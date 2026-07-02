from __future__ import annotations

import math
import time
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


class GSLRUStrategy(EvictionStrategy):
    """Graduated SLRU with top-segment decay boost: tau=20 for max segment,
    tau=decay_tau for all others. V31 optimum (device_weight=5, tau_top=20)."""

    def __init__(self, max_segment: int = 4, decay_tau: float = 15.0):
        self.max_segment = max_segment
        self.decay_tau = decay_tau
        self.decay_tau_top = decay_tau * 4.0 / 3.0

    def get_priority(self, node: TreeNode) -> Tuple[float, float]:
        segment = min(node.hit_count, self.max_segment)
        if self.decay_tau > 0:
            age = time.monotonic() - node.last_access_time
            tau = self.decay_tau_top if segment == self.max_segment else self.decay_tau
            decay = math.exp(-age / tau)
            return (segment * decay, node.last_access_time)
        return (float(segment), node.last_access_time)
