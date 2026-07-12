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


class CARStrategy(EvictionStrategy):
    """Continuation-Aware Residency (wilkes). Combines hit-count segmentation (protect PROVEN multi-turn
    conversations indefinitely, like SLRU — handles long think-gaps) with a completion GRACE probation for
    UNPROVEN turn-0s (protect a just-completed conversation for grace-s ~ the think-gap, so its FIRST reuse
    turn-0->turn-1 lands before eviction — the race hit-count policies miss, since turn-0 has hit_count=0).
    Single-turn "whale" documents complete once, are never reused, and their grace expires -> evicted first,
    freeing capacity for proven convs. Soft: LRU within each segment, so eviction always makes progress.

    Priority (heap pops MIN = evict first):
      segment 0 (evict first): unproven (hit_count==0) AND grace expired  -> LRU  [single-turn whales]
      segment 1:               unproven (hit_count==0) but within grace   -> LRU  [turn-0 probation]
      segment 2 (protect most):proven (hit_count>=1)                       -> LRU  [active multi-turn]
    """

    def __init__(self, grace_s: float = 30.0):
        self.grace_s = grace_s

    def get_priority(self, node: TreeNode) -> Tuple[int, float]:
        import time as _t
        if getattr(node, "hit_count", 0) >= 1:
            seg = 2
        elif (_t.time() - getattr(node, "car_completed_at", 0.0)) <= self.grace_s:
            seg = 1
        else:
            seg = 0
        return (seg, node.last_access_time)


class WhaleStrategy(EvictionStrategy):
    """Size-aware liveness eviction (wilkes). Exploits the measured turn-0-SIZE => continuation signal:
    single-turn "whale" documents have a LARGE turn-0 (median ~17K tok) while multi-turn conversations start
    SMALL (median ~816 tok) — AUC(turn-0 size -> continuation) ~= 0.78. So among UNPROVEN nodes (hit_count==0),
    the LARGEST are the likeliest single-turn whales (safe to evict: never reused), and the smallest are the
    likeliest multi-turn turn-0s (protect: first reuse is coming). A CAUSAL proxy for Belady's evict-dead-first,
    using an observable turn-0 feature (size) instead of the future.

    Priority (heap pops MIN = evict first):
      unproven (hit_count==0): (0, -size, last_access) -> evict LARGEST unproven (whale) first, LRU tiebreak
      proven   (hit_count>=1): (1, 0, last_access)     -> protect; plain LRU within the proven segment
    Rationale vs SLRU: SLRU evicts unproven by RECENCY (sacrificing recent small turn-0 continuers => it HURT);
    Whale evicts unproven by SIZE (dropping big whales, sparing small continuers)."""

    def get_priority(self, node: TreeNode) -> Tuple:
        if node.hit_count >= 1:
            return (1, 0.0, node.last_access_time)
        size = float(len(node.key)) if node.key is not None else 0.0
        return (0, -size, node.last_access_time)


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
