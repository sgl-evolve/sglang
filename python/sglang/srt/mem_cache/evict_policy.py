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
    """Continuation-Aware Residency (wilkes). A conversation's prefix is protected from eviction for a
    grace window (wall-seconds ~ the observed think-gap) after each request completes on it, so it
    survives the idle gap until the next turn reuses it — capturing the turn-0->turn-1 first-reuse race
    that hit-count policies (SLRU/LFU) miss (turn-0 has hit_count=0). Multi-turn convs are reused within
    grace (re-pinned, kept); single-turn "whale" documents' grace expires and they are evicted. Soft:
    within each segment falls back to LRU, so eviction always makes progress under full pressure.

    Priority (heap pops MIN = evict first):
      segment 0 (evict first): grace expired / never completed -> ordered by last_access_time (LRU)
      segment 1 (protect):     completed within grace           -> ordered by last_access_time (LRU)
    """

    def __init__(self, grace_s: float = 30.0):
        self.grace_s = grace_s

    def get_priority(self, node: TreeNode) -> Tuple[int, float]:
        import time as _t
        protected = (_t.time() - getattr(node, "car_completed_at", 0.0)) <= self.grace_s
        return (1 if protected else 0, node.last_access_time)


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
