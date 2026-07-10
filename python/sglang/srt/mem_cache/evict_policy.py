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

    def __init__(
        self,
        threshold: int = 4096,
        reuse_min: int = 0,
        threshold2: int = 0,
        cost_mode: str = "segment",
    ):
        self.threshold = threshold
        self.reuse_min = reuse_min
        self.threshold2 = threshold2
        # "segment" = recompute cost of THIS node's own tokens (len(key)); "depth" = cumulative prefix
        # length from root (protects deep conversation tails — short late turns of long multiturn chats
        # that segment-mode misses). depth targets the charter's conversation-co-residency goal directly.
        self.cost_mode = cost_mode

    def get_priority(self, node: TreeNode) -> Tuple[int, float]:
        key = getattr(node, "key", None)
        seg = len(key) if key is not None else 0
        if self.cost_mode == "depth":
            # max() is a safety net: prefix_len >= own segment always, so this uses the maintained
            # cumulative depth when set and never falls below segment cost if a node missed maintenance.
            cost = max(getattr(node, "prefix_len", 0), seg)
        else:
            cost = seg
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


class GDSFStrategy(EvictionStrategy):
    """Greedy-Dual-Size-Frequency eviction (web-caching classic).

    Combines frequency, cost (recompute = prefix len), and size (node token count)
    into a single priority: H(p) = freq * cost / size + aging_clock. Lower H →
    evicted first. Unlike pure LFU this accounts for the VALUE of retaining a node
    (high cost, small size, frequently hit = most valuable). The aging clock prevents
    stale high-frequency nodes from being immortal (cache pollution).

    For KV-cache: cost = segment length (recompute tokens), size = len(key) (page
    tokens consumed), freq = hit_count. A long shared document prefix that's hit
    often gets high priority; a one-shot short leaf gets low priority.
    """

    def get_priority(self, node: TreeNode) -> float:
        key = getattr(node, "key", None)
        size = max(len(key), 1) if key is not None else 1
        cost = getattr(node, "prefix_len", 0) or size
        freq = max(getattr(node, "hit_count", 0), 1)
        # Higher value = retained longer (heap pops minimum = evicted first).
        # Use last_access_time as the aging clock component.
        return freq * cost / size + node.last_access_time


class ContinuousCostStrategy(EvictionStrategy):
    """Continuous recompute-cost-weighted LRU (no threshold discretization).

    Instead of segmenting into discrete tiers (cheap/expensive), this uses a
    continuous cost weight: priority = last_access_time + alpha * log(1 + cost).
    Higher priority = retained longer; the log dampens the cost influence so it
    acts as a tiebreaker within a recency window rather than overriding recency
    entirely. alpha controls the cost-vs-recency tradeoff.
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def get_priority(self, node: TreeNode) -> float:
        import math

        key = getattr(node, "key", None)
        seg = len(key) if key is not None else 0
        return node.last_access_time + self.alpha * math.log1p(seg)


class CostFreqStrategy(EvictionStrategy):
    """Cost × frequency hybrid: protect nodes that are BOTH expensive AND frequently hit.

    Combines the cost-aware insight (long prefixes are expensive to recompute) with
    frequency awareness (frequently-hit nodes are more likely to be reused). Segments
    by cost (like CostAwareStrategy) but within the expensive tier, uses frequency
    to further prioritize — a frequently-hit expensive node is evicted LAST.
    """

    def __init__(self, threshold: int = 2048):
        self.threshold = threshold

    def get_priority(self, node: TreeNode) -> Tuple:
        key = getattr(node, "key", None)
        seg = len(key) if key is not None else 0
        tier = 1 if seg >= self.threshold else 0
        freq = getattr(node, "hit_count", 0)
        # Within each tier: higher freq → retained longer, then LRU within same freq
        return (tier, freq, node.last_access_time)


class FreqDecayStrategy(EvictionStrategy):
    """LFU with exponential time-decay on frequency, preventing cache pollution.

    Pure LFU suffers from stale-high-frequency nodes blocking eviction long after
    they stop being useful. This decays frequency by age: effective_freq =
    hit_count * decay^(age_seconds), where decay < 1. Recent high-frequency nodes
    keep high priority; old ones decay away. Falls back to recency when
    frequencies are similar.
    """

    def __init__(self, decay: float = 0.999):
        self.decay = decay

    def get_priority(self, node: TreeNode) -> float:
        import time as _time

        freq = max(getattr(node, "hit_count", 0), 1)
        now = _time.monotonic()
        age = max(0, now - node.last_access_time)
        decayed = freq * (self.decay ** age)
        return decayed + node.last_access_time * 1e-12


class SizeAwareLRUStrategy(EvictionStrategy):
    """Size-aware LRU: prefer evicting larger nodes (they free more capacity per eviction).

    Under memory pressure, evicting one large node recovers more tokens than many
    small ones, reducing eviction overhead and churn. priority = (size_bucket,
    last_access_time): among equally-old nodes, larger ones are evicted first
    (they have lower priority). size_bucket inverts size into priority: bucket 0 =
    large (evict first), bucket 1 = small (evict later).
    """

    def __init__(self, size_threshold: int = 2048):
        self.size_threshold = size_threshold

    def get_priority(self, node: TreeNode) -> Tuple[int, float]:
        key = getattr(node, "key", None)
        seg = len(key) if key is not None else 0
        # Large nodes (>= threshold) → bucket 0 (evict first); small → bucket 1
        bucket = 0 if seg >= self.size_threshold else 1
        return (bucket, node.last_access_time)
