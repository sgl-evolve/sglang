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
    """Recompute-cost-aware eviction: protects nodes whose prefix would be
    expensive to recompute.  A node's 'cost' is approximated by its depth in
    the radix tree (prefix_depth * num_tokens) — deeper, longer nodes represent
    more cumulative prefill work.  We evict the cheapest nodes first (lowest
    cost/recency product), so large shared prefixes survive longer.
    Env: SGLANG_COSTEVICT_THRESHOLD (int, default 2048) — nodes covering
    fewer cumulative tokens than this are treated as zero-cost (plain LRU)."""

    def __init__(self):
        import os

        self.threshold = int(os.environ.get("SGLANG_COSTEVICT_THRESHOLD", "2048"))

    def get_priority(self, node: TreeNode) -> Tuple[float, float]:
        cost = self._node_cost(node)
        tier = 0 if cost < self.threshold else 1
        return (tier, node.last_access_time)

    @staticmethod
    def _node_cost(node: TreeNode) -> int:
        cost = 0
        cur = node
        while cur.parent is not None:
            cost += len(cur.key.token_ids) if cur.key is not None else 64
            cur = cur.parent
        return cost

    @staticmethod
    def _node_size(node: TreeNode) -> int:
        if node.key is not None:
            return max(1, len(node.key.token_ids))
        return 64


class GDSFStrategy(EvictionStrategy):
    """Greedy-Dual-Size-Frequency: evict by cost/(size*freq) — nodes that are
    large, cheap to recompute, and rarely accessed get evicted first.
    Approximation: cost = prefix depth tokens, size = node tokens, freq = hit_count+1."""

    def get_priority(self, node: TreeNode) -> Tuple[float, float]:
        size = CostAwareStrategy._node_size(node)
        freq = max(1, node.hit_count + 1)
        cost = CostAwareStrategy._node_cost(node)
        score = (cost * freq) / size
        return (score, node.last_access_time)
