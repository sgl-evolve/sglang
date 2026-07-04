"""Unit tests for evict_policy.py"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")
register_cpu_ci(est_time=7, suite="base-c-test-cpu")

import unittest
from unittest.mock import MagicMock

from sglang.srt.mem_cache.evict_policy import (
    CostAwareStrategy,
    CostFreqStrategy,
    CostTieredStrategy,
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


def _make_depth_node(depth, last_access_time=0.0, hit_count=0, seg=None):
    """Build a leaf whose cumulative prefix DEPTH (tokens root->node) == `depth`.

    The cost-aware strategies walk parents summing len(node.key); `seg` optionally
    splits the depth across a multi-hop chain to exercise the bounded parent-walk.
    """
    root = MagicMock()
    root.key = None
    root.parent = None
    seg = seg or max(depth, 1)
    node = None
    remaining = depth
    child = root
    while remaining > 0:
        take = min(seg, remaining)
        n = MagicMock()
        n.key = list(range(take))
        n.parent = child
        child = n
        node = n
        remaining -= take
    if node is None:  # depth == 0
        node = MagicMock()
        node.key = []
        node.parent = root
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
    """Recompute-cost-aware eviction: protect DEEP (expensive-to-recompute) prefixes;
    evict SHALLOW (cheap) first; LRU within each depth bucket. Threshold = 8192 tokens."""

    def setUp(self):
        self.strategy = CostAwareStrategy()

    def test_shallow_is_bucket_zero(self):
        node = _make_depth_node(depth=2000, last_access_time=5.0)
        self.assertEqual(self.strategy.get_priority(node), (0, 5.0))

    def test_deep_is_bucket_one(self):
        node = _make_depth_node(depth=9000, last_access_time=5.0)
        self.assertEqual(self.strategy.get_priority(node), (1, 5.0))

    def test_at_threshold_is_deep(self):
        node = _make_depth_node(depth=8192, last_access_time=5.0)
        self.assertEqual(self.strategy.get_priority(node), (1, 5.0))

    def test_shallow_evicted_before_deep(self):
        shallow = _make_depth_node(depth=1000, last_access_time=100.0)  # even if recent
        deep = _make_depth_node(depth=20000, last_access_time=1.0)  # even if old
        self.assertLess(
            self.strategy.get_priority(shallow), self.strategy.get_priority(deep)
        )

    def test_lru_within_bucket(self):
        old = _make_depth_node(depth=9000, last_access_time=1.0)
        new = _make_depth_node(depth=9000, last_access_time=10.0)
        self.assertLess(
            self.strategy.get_priority(old), self.strategy.get_priority(new)
        )

    def test_depth_via_multihop_chain(self):
        # depth split into 64-token segments (page-aligned) still classifies deep
        node = _make_depth_node(depth=9000, last_access_time=3.0, seg=64)
        self.assertEqual(self.strategy.get_priority(node), (1, 3.0))

    def test_walk_is_bounded(self):
        # Very deep prefix in tiny segments must still terminate (break at threshold).
        node = _make_depth_node(depth=40000, last_access_time=2.0, seg=64)
        self.assertEqual(self.strategy.get_priority(node), (1, 2.0))


class TestCostTieredStrategy(unittest.TestCase):
    """Graded depth tiers (tier=depth//8192 capped at 4): deeper -> higher tier -> kept longer."""

    def setUp(self):
        self.strategy = CostTieredStrategy()

    def test_tier_increases_with_depth(self):
        t0 = self.strategy.get_priority(_make_depth_node(depth=1000))[0]
        t1 = self.strategy.get_priority(_make_depth_node(depth=9000))[0]
        t2 = self.strategy.get_priority(_make_depth_node(depth=17000))[0]
        self.assertEqual((t0, t1, t2), (0, 1, 2))

    def test_tier_capped(self):
        big = self.strategy.get_priority(_make_depth_node(depth=100000, seg=8192))[0]
        self.assertEqual(big, self.strategy.MAX_TIER)

    def test_lower_tier_evicted_first(self):
        shallow = _make_depth_node(depth=1000, last_access_time=100.0)
        deep = _make_depth_node(depth=40000, last_access_time=1.0)
        self.assertLess(
            self.strategy.get_priority(shallow), self.strategy.get_priority(deep)
        )


class TestCostFreqStrategy(unittest.TestCase):
    """2-factor: (is_deep, hit_bucket, last_access). Protect deep AND frequently-reused."""

    def setUp(self):
        self.strategy = CostFreqStrategy()

    def test_priority_tuple_shape(self):
        node = _make_depth_node(depth=9000, last_access_time=4.0, hit_count=3)
        self.assertEqual(self.strategy.get_priority(node), (1, 3, 4.0))

    def test_hit_count_capped(self):
        node = _make_depth_node(depth=1000, last_access_time=4.0, hit_count=999)
        self.assertEqual(
            self.strategy.get_priority(node), (0, self.strategy.HIT_CAP, 4.0)
        )

    def test_depth_dominates_frequency(self):
        # a shallow, very-hot node is still evicted before a deep, cold node
        shallow_hot = _make_depth_node(depth=1000, last_access_time=1.0, hit_count=100)
        deep_cold = _make_depth_node(depth=20000, last_access_time=100.0, hit_count=0)
        self.assertLess(
            self.strategy.get_priority(shallow_hot),
            self.strategy.get_priority(deep_cold),
        )

    def test_frequency_tiebreak_within_bucket(self):
        # within the same depth bucket, the less-frequently-hit is evicted first
        cold = _make_depth_node(depth=9000, last_access_time=10.0, hit_count=1)
        hot = _make_depth_node(depth=9000, last_access_time=1.0, hit_count=5)
        self.assertLess(
            self.strategy.get_priority(cold), self.strategy.get_priority(hot)
        )


if __name__ == "__main__":
    unittest.main()
