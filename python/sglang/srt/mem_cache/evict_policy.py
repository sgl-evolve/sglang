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

    DEPTH_THRESHOLD = 8192  # tokens; swept peak (separates short turns from long-context prefixes)
    MAX_WALK = 512          # cap the parent walk so get_priority stays ~O(1)

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


class CostFreqStrategy(EvictionStrategy):
    """2-factor cost×frequency eviction: like CostAwareStrategy (protect DEEP/expensive prefixes,
    threshold 8192), but WITHIN each depth bucket, protect FREQUENTLY-REUSED nodes (higher hit_count)
    before evicting by recency. Rationale: cost-aware alone can evict a shallow-but-hot prefix (a
    short doc reused by many questions) that is cheap-per-hit but reused often. Ordering:
    evict (shallow, low-hit, old) first; keep (deep, high-hit, recent). Lossless (eviction only).
    """

    DEPTH_THRESHOLD = 8192
    MAX_WALK = 512
    HIT_CAP = 8  # cap hit_count influence so a few mega-hit nodes don't dominate

    def get_priority(self, node: TreeNode) -> Tuple[int, int, float]:
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
        hit_bucket = min(getattr(node, "hit_count", 0), self.HIT_CAP)
        # min-heap: (shallow, infrequent, old) evicted first; (deep, frequent, recent) kept
        return (is_deep, hit_bucket, node.last_access_time)


class CostTieredStrategy(EvictionStrategy):
    """Finer-grained cost-aware eviction: instead of the binary deep/shallow bucket of
    CostAwareStrategy (peak at threshold 8192), assign a graded DEPTH TIER so the truly-huge
    prefixes (LEval/LooGLE 100k+, most O(L^2)-expensive to recompute) are protected MORE than
    the moderately-long. tier = min(depth // TIER_SIZE, MAX_TIER); evict lowest tier first,
    LRU within a tier. Same losslessness + bounded parent-walk as CostAwareStrategy.
    """

    TIER_SIZE = 8192   # tokens per protection tier (aligns with the swept binary peak)
    MAX_TIER = 4       # cap: tiers 0..4 (>=32768 tokens all top-tier); bounds the walk
    MAX_WALK = 512

    def get_priority(self, node: TreeNode) -> Tuple[int, float]:
        depth = 0
        n = node
        hops = 0
        cap = self.TIER_SIZE * self.MAX_TIER
        while n is not None and getattr(n, "key", None) is not None and hops < self.MAX_WALK:
            depth += len(n.key)
            if depth >= cap:
                break
            n = n.parent
            hops += 1
        tier = min(depth // self.TIER_SIZE, self.MAX_TIER)
        # min-heap pops smallest first: (low tier=shallow, older) evicted before (high tier=deep)
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
