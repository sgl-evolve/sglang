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


class CostAwareStrategy(EvictionStrategy):
    """Recompute-cost-aware eviction for the best_effort (recompute-on-miss) regime.

    Under best_effort a miss is recomputed on the GPU; prefill cost is dominated by the
    O(L^2) attention term for LONG prefixes (the LEval/LooGLE portion of the mix, 100k+
    tokens), even though this is a mamba-hybrid (only ~12/48 layers are attention). So a
    long/deep prefix is far more expensive to recompute than a short one. LRU ignores this.

    Policy: bucket leaves by prefix DEPTH (cumulative tokens root->node) — shallow (cheap to
    recompute) evicted before deep (expensive) — then LRU within each bucket. Deep prefixes
    are retained longer, trading a few cheap recomputes for avoiding expensive O(L^2) ones.

    Lossless: eviction only changes WHAT is cached; a wrongly-evicted node just recomputes
    (exactly what best_effort already does). Depth is computed by a bounded parent-walk so a
    pathological tree can't make eviction quadratic.
    """

    DEPTH_THRESHOLD = 4096  # tokens; separates short turns from long-context prefixes
    MAX_WALK = 96           # cap the parent walk so get_priority stays ~O(1)

    def get_priority(self, node: TreeNode) -> Tuple[int, float]:
        depth = 0
        n = node
        hops = 0
        while n is not None and getattr(n, "key", None) is not None and hops < self.MAX_WALK:
            depth += len(n.key)
            if depth >= self.DEPTH_THRESHOLD:
                break
            n = n.parent
            hops += 1
        is_deep = 1 if depth >= self.DEPTH_THRESHOLD else 0
        # min-heap pops smallest first: (shallow=0, older) evicted before (deep=1, ...)
        return (is_deep, node.last_access_time)


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
