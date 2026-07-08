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
    """Recompute-cost-aware eviction for tail-latency SLOs.

    Motivation: in a tiered KV cache under memory pressure, count-optimal
    replacement (LRU, which ~ matches Belady for in-order reuse) minimizes the
    *number* of misses but not their *cost*. Per-miss recompute cost is ~ the
    lost prefix length, which spans orders of magnitude (a 1k chat turn vs a
    128k document prefix). Under an SLO the tail is what matters, and the tail
    is dominated by the few catastrophic long-prefix recomputes. This strategy
    segments evictable nodes by recompute cost (prefix segment length) and
    evicts *cheap* segments before *expensive* ones, LRU within each segment —
    so long, expensive-to-recompute prefixes are retained longer and their
    misses (which drive p99 TTFT) are avoided, at the price of more cheap
    misses (which barely move the tail). Lossless: only eviction order changes.

    priority is a (segment, last_access_time) tuple; the eviction heap pops the
    minimum, so segment 0 (cheap) is evicted before segment 1 (expensive), and
    older nodes go first within a segment.

    Reuse gating (``reuse_min`` > 0): only protect a prefix that is BOTH expensive
    AND has proven reuse (``hit_count`` >= reuse_min, i.e. it was re-matched by a
    later request). This avoids sacrificing reused short prefixes to protect
    one-shot long prefixes (e.g. a single long-context query with no follow-up),
    which is the main source of the median-TTFT regression from pure cost gating.
    reuse_min = 0 recovers the pure cost-aware behavior.

    Multi-tier segmentation (``threshold2`` > 0): use TWO cost boundaries so the
    *longest* (most catastrophic-to-recompute, p99-tail-driving) prefixes are
    protected MORE strongly than merely-long ones. Tier = number of thresholds
    the cost meets/exceeds: short (<threshold) → tier 0 (evicted first), long
    (threshold..threshold2) → tier 1, longest (>=threshold2) → tier 2 (evicted
    last). Unlike lowering a single threshold (which protects *more* prefixes and
    can over-protect / worsen the tail), this keeps the same protected set but
    orders it by cost, targeting the tail. threshold2 = 0 recovers 2-tier.
    """

    def __init__(self, threshold: int = 4096, reuse_min: int = 0, threshold2: int = 0):
        self.threshold = threshold
        self.reuse_min = reuse_min
        self.threshold2 = threshold2

    def get_priority(self, node: TreeNode) -> Tuple[int, float]:
        key = getattr(node, "key", None)
        cost = len(key) if key is not None else 0
        tier = 1 if cost >= self.threshold else 0
        if tier == 1 and self.reuse_min > 0:
            # unproven long prefixes fall back to the cheap segment
            if getattr(node, "hit_count", 0) < self.reuse_min:
                tier = 0
        if tier >= 1 and self.threshold2 > 0 and cost >= self.threshold2:
            tier = 2
        return (tier, node.last_access_time)


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
