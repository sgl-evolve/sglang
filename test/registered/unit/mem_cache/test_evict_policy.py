"""Unit tests for evict_policy.py"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")
register_cpu_ci(est_time=7, suite="base-c-test-cpu")

import unittest
from unittest.mock import MagicMock

from sglang.srt.mem_cache.evict_policy import (
    CostAwareStrategy,
    FIFOStrategy,
    FILOStrategy,
    LFUStrategy,
    LRUStrategy,
    MRUStrategy,
    PriorityStrategy,
    SLRUStrategy,
)


def _make_node(**kwargs):
    node = MagicMock()
    node.last_access_time = kwargs.get("last_access_time", 0.0)
    node.hit_count = kwargs.get("hit_count", 0)
    node.creation_time = kwargs.get("creation_time", 0.0)
    node.priority = kwargs.get("priority", 0)
    return node


class _Key:
    """Minimal RadixKey stand-in: only __len__ is used by CostAwareStrategy."""

    def __init__(self, n):
        self._n = n

    def __len__(self):
        return self._n


def _make_cost_node(seg_len, prefix_len=None, last_access_time=0.0, hit_count=1):
    """Node for CostAwareStrategy: has key (segment length), prefix_len (depth)."""
    node = MagicMock()
    node.key = _Key(seg_len)
    node.prefix_len = seg_len if prefix_len is None else prefix_len
    node.last_access_time = last_access_time
    node.hit_count = hit_count
    return node


class TestLRUStrategy(unittest.TestCase):
    def setUp(self):
        self.strategy = LRUStrategy()

    def test_priority_is_last_access_time(self):
        node = _make_node(last_access_time=42.0)
        self.assertEqual(self.strategy.get_priority(node), 42.0)

    def test_older_access_evicted_first(self):
        old = _make_node(last_access_time=1.0)
        new = _make_node(last_access_time=10.0)
        self.assertLess(
            self.strategy.get_priority(old), self.strategy.get_priority(new)
        )


class TestLFUStrategy(unittest.TestCase):
    def setUp(self):
        self.strategy = LFUStrategy()

    def test_priority_is_hit_count_and_time(self):
        node = _make_node(hit_count=5, last_access_time=3.0)
        self.assertEqual(self.strategy.get_priority(node), (5, 3.0))

    def test_lower_hit_count_evicted_first(self):
        cold = _make_node(hit_count=1, last_access_time=10.0)
        hot = _make_node(hit_count=100, last_access_time=1.0)
        self.assertLess(
            self.strategy.get_priority(cold), self.strategy.get_priority(hot)
        )

    def test_same_hit_count_older_access_evicted_first(self):
        old = _make_node(hit_count=3, last_access_time=1.0)
        new = _make_node(hit_count=3, last_access_time=10.0)
        self.assertLess(
            self.strategy.get_priority(old), self.strategy.get_priority(new)
        )


class TestFIFOStrategy(unittest.TestCase):
    def setUp(self):
        self.strategy = FIFOStrategy()

    def test_priority_is_creation_time(self):
        node = _make_node(creation_time=7.0)
        self.assertEqual(self.strategy.get_priority(node), 7.0)

    def test_earlier_created_evicted_first(self):
        first = _make_node(creation_time=1.0)
        second = _make_node(creation_time=5.0)
        self.assertLess(
            self.strategy.get_priority(first), self.strategy.get_priority(second)
        )


class TestMRUStrategy(unittest.TestCase):
    def setUp(self):
        self.strategy = MRUStrategy()

    def test_priority_is_negated_access_time(self):
        node = _make_node(last_access_time=5.0)
        self.assertEqual(self.strategy.get_priority(node), -5.0)

    def test_most_recently_used_evicted_first(self):
        """MRU evicts the most recently accessed node first (lowest priority value)."""
        old = _make_node(last_access_time=1.0)
        new = _make_node(last_access_time=10.0)
        self.assertLess(
            self.strategy.get_priority(new), self.strategy.get_priority(old)
        )


class TestFILOStrategy(unittest.TestCase):
    def setUp(self):
        self.strategy = FILOStrategy()

    def test_priority_is_negated_creation_time(self):
        node = _make_node(creation_time=3.0)
        self.assertEqual(self.strategy.get_priority(node), -3.0)

    def test_last_created_evicted_first(self):
        """FILO evicts the most recently created node first."""
        first = _make_node(creation_time=1.0)
        second = _make_node(creation_time=5.0)
        self.assertLess(
            self.strategy.get_priority(second), self.strategy.get_priority(first)
        )


class TestPriorityStrategy(unittest.TestCase):
    def setUp(self):
        self.strategy = PriorityStrategy()

    def test_priority_is_tuple(self):
        node = _make_node(priority=2, last_access_time=4.0)
        self.assertEqual(self.strategy.get_priority(node), (2, 4.0))

    def test_lower_priority_evicted_first(self):
        low = _make_node(priority=1, last_access_time=10.0)
        high = _make_node(priority=5, last_access_time=1.0)
        self.assertLess(
            self.strategy.get_priority(low), self.strategy.get_priority(high)
        )

    def test_same_priority_older_access_evicted_first(self):
        old = _make_node(priority=3, last_access_time=1.0)
        new = _make_node(priority=3, last_access_time=10.0)
        self.assertLess(
            self.strategy.get_priority(old), self.strategy.get_priority(new)
        )


class TestSLRUStrategy(unittest.TestCase):
    def setUp(self):
        self.strategy = SLRUStrategy(protected_threshold=2)

    def test_probationary_segment(self):
        node = _make_node(hit_count=1, last_access_time=5.0)
        self.assertEqual(self.strategy.get_priority(node), (0, 5.0))

    def test_protected_segment(self):
        node = _make_node(hit_count=2, last_access_time=5.0)
        self.assertEqual(self.strategy.get_priority(node), (1, 5.0))

    def test_highly_accessed_is_protected(self):
        node = _make_node(hit_count=100, last_access_time=5.0)
        self.assertEqual(self.strategy.get_priority(node), (1, 5.0))

    def test_probationary_evicted_before_protected(self):
        prob = _make_node(hit_count=1, last_access_time=10.0)
        prot = _make_node(hit_count=5, last_access_time=1.0)
        self.assertLess(
            self.strategy.get_priority(prob), self.strategy.get_priority(prot)
        )

    def test_same_segment_older_access_evicted_first(self):
        old = _make_node(hit_count=0, last_access_time=1.0)
        new = _make_node(hit_count=0, last_access_time=10.0)
        self.assertLess(
            self.strategy.get_priority(old), self.strategy.get_priority(new)
        )

    def test_custom_threshold(self):
        strategy = SLRUStrategy(protected_threshold=5)
        below = _make_node(hit_count=4, last_access_time=1.0)
        at = _make_node(hit_count=5, last_access_time=1.0)
        self.assertEqual(strategy.get_priority(below), (0, 1.0))
        self.assertEqual(strategy.get_priority(at), (1, 1.0))

    def test_default_threshold_is_2(self):
        default = SLRUStrategy()
        self.assertEqual(default.protected_threshold, 2)


class TestEvictionOrdering(unittest.TestCase):
    """Integration-style test: sort a list of nodes by eviction priority."""

    def test_lru_ordering(self):
        strategy = LRUStrategy()
        nodes = [
            _make_node(last_access_time=5.0),
            _make_node(last_access_time=1.0),
            _make_node(last_access_time=3.0),
        ]
        eviction_order = sorted(nodes, key=strategy.get_priority)
        times = [n.last_access_time for n in eviction_order]
        self.assertEqual(times, [1.0, 3.0, 5.0])

    def test_slru_ordering(self):
        strategy = SLRUStrategy(protected_threshold=2)
        nodes = [
            _make_node(hit_count=5, last_access_time=1.0),  # protected, old
            _make_node(hit_count=0, last_access_time=10.0),  # probationary, new
            _make_node(hit_count=0, last_access_time=2.0),  # probationary, old
            _make_node(hit_count=3, last_access_time=8.0),  # protected, new
        ]
        eviction_order = sorted(nodes, key=strategy.get_priority)
        expected = [
            (0, 2.0),  # probationary old
            (0, 10.0),  # probationary new
            (1, 1.0),  # protected old
            (1, 8.0),  # protected new
        ]
        actual = [strategy.get_priority(n) for n in eviction_order]
        self.assertEqual(actual, expected)


class TestCostAwareStrategy(unittest.TestCase):
    """Recompute-cost-aware eviction: cheap (short) prefixes evicted before
    expensive (long) ones, LRU within a cost tier. Lossless — order only."""

    def setUp(self):
        self.strategy = CostAwareStrategy(threshold=2048)

    def test_priority_is_tier_and_access_time(self):
        node = _make_cost_node(seg_len=100, last_access_time=4.0)
        self.assertEqual(self.strategy.get_priority(node), (0, 4.0))

    def test_cheap_segment_below_threshold(self):
        node = _make_cost_node(seg_len=2047, last_access_time=5.0)
        self.assertEqual(self.strategy.get_priority(node), (0, 5.0))

    def test_expensive_segment_at_threshold(self):
        # >= threshold is protected (tier 1).
        node = _make_cost_node(seg_len=2048, last_access_time=5.0)
        self.assertEqual(self.strategy.get_priority(node), (1, 5.0))

    def test_cheap_evicted_before_expensive(self):
        """A recently-used cheap prefix is still evicted before an older
        expensive one — the whole point of cost-aware retention."""
        cheap_recent = _make_cost_node(seg_len=100, last_access_time=100.0)
        expensive_old = _make_cost_node(seg_len=8192, last_access_time=1.0)
        self.assertLess(
            self.strategy.get_priority(cheap_recent),
            self.strategy.get_priority(expensive_old),
        )

    def test_same_tier_older_access_evicted_first(self):
        old = _make_cost_node(seg_len=100, last_access_time=1.0)
        new = _make_cost_node(seg_len=200, last_access_time=10.0)
        self.assertLess(
            self.strategy.get_priority(old), self.strategy.get_priority(new)
        )

    def test_missing_key_treated_as_zero_cost(self):
        node = MagicMock()
        node.key = None
        node.prefix_len = 0
        node.last_access_time = 3.0
        node.hit_count = 0
        self.assertEqual(self.strategy.get_priority(node), (0, 3.0))

    def test_default_threshold_is_4096(self):
        default = CostAwareStrategy()
        self.assertEqual(default.threshold, 4096)
        # 4095 cheap, 4096 protected under the default.
        self.assertEqual(
            default.get_priority(_make_cost_node(seg_len=4095)), (0, 0.0)
        )
        self.assertEqual(
            default.get_priority(_make_cost_node(seg_len=4096)), (1, 0.0)
        )


class TestCostAwareReuseGating(unittest.TestCase):
    """reuse_min > 0: only protect a long prefix that ALSO has proven reuse."""

    def setUp(self):
        self.strategy = CostAwareStrategy(threshold=2048, reuse_min=1)

    def test_unproven_expensive_prefix_falls_back_to_cheap(self):
        # long but never re-matched (hit_count 0) -> demoted to the cheap tier.
        node = _make_cost_node(seg_len=8192, last_access_time=5.0, hit_count=0)
        self.assertEqual(self.strategy.get_priority(node), (0, 5.0))

    def test_proven_expensive_prefix_is_protected(self):
        node = _make_cost_node(seg_len=8192, last_access_time=5.0, hit_count=1)
        self.assertEqual(self.strategy.get_priority(node), (1, 5.0))

    def test_cheap_prefix_unaffected_by_reuse_gating(self):
        node = _make_cost_node(seg_len=100, last_access_time=5.0, hit_count=0)
        self.assertEqual(self.strategy.get_priority(node), (0, 5.0))


class TestCostAwareThreeTier(unittest.TestCase):
    """threshold2 > 0: the longest prefixes are protected MORE than merely-long
    ones (tier 2 > tier 1 > tier 0), targeting the p99 tail without enlarging
    the protected set."""

    def setUp(self):
        self.strategy = CostAwareStrategy(threshold=2048, threshold2=16384)

    def test_short_is_tier0(self):
        node = _make_cost_node(seg_len=100, last_access_time=1.0)
        self.assertEqual(self.strategy.get_priority(node), (0, 1.0))

    def test_long_is_tier1(self):
        node = _make_cost_node(seg_len=4096, last_access_time=1.0)
        self.assertEqual(self.strategy.get_priority(node), (1, 1.0))

    def test_longest_is_tier2(self):
        node = _make_cost_node(seg_len=32768, last_access_time=1.0)
        self.assertEqual(self.strategy.get_priority(node), (2, 1.0))

    def test_eviction_order_short_long_longest(self):
        nodes = [
            _make_cost_node(seg_len=32768, last_access_time=1.0),  # tier 2
            _make_cost_node(seg_len=100, last_access_time=1.0),  # tier 0
            _make_cost_node(seg_len=4096, last_access_time=1.0),  # tier 1
        ]
        order = sorted(nodes, key=self.strategy.get_priority)
        tiers = [self.strategy.get_priority(n)[0] for n in order]
        self.assertEqual(tiers, [0, 1, 2])


class TestCostAwareDepthMode(unittest.TestCase):
    """depth mode: cost = cumulative prefix length (protects short late turns of
    long multiturn chats that segment mode would treat as cheap)."""

    def setUp(self):
        self.strategy = CostAwareStrategy(threshold=2048, cost_mode="depth")

    def test_deep_short_segment_is_protected(self):
        # own segment is small (a short late turn) but it sits deep in a long
        # conversation -> depth mode protects it, segment mode would not.
        node = _make_cost_node(seg_len=100, prefix_len=9000, last_access_time=1.0)
        self.assertEqual(self.strategy.get_priority(node), (1, 1.0))

    def test_segment_mode_would_not_protect_same_node(self):
        seg_strategy = CostAwareStrategy(threshold=2048, cost_mode="segment")
        node = _make_cost_node(seg_len=100, prefix_len=9000, last_access_time=1.0)
        self.assertEqual(seg_strategy.get_priority(node), (0, 1.0))

    def test_depth_never_below_segment_cost(self):
        # prefix_len maintenance miss (stale 0): max() falls back to segment len.
        node = _make_cost_node(seg_len=4096, prefix_len=0, last_access_time=1.0)
        self.assertEqual(self.strategy.get_priority(node), (1, 1.0))


class TestCostAwareOrdering(unittest.TestCase):
    """Integration-style: sort a mixed set by cost-aware eviction priority."""

    def test_ordering_cheap_first_lru_within_tier(self):
        strategy = CostAwareStrategy(threshold=2048)
        nodes = [
            _make_cost_node(seg_len=8192, last_access_time=2.0),  # expensive, old
            _make_cost_node(seg_len=8192, last_access_time=9.0),  # expensive, new
            _make_cost_node(seg_len=50, last_access_time=3.0),  # cheap, old
            _make_cost_node(seg_len=50, last_access_time=8.0),  # cheap, new
        ]
        order = sorted(nodes, key=strategy.get_priority)
        actual = [strategy.get_priority(n) for n in order]
        self.assertEqual(actual, [(0, 3.0), (0, 8.0), (1, 2.0), (1, 9.0)])


if __name__ == "__main__":
    unittest.main()
