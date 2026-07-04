"""Unit test for the SJF (shortest-uncached-prefill-first) schedule policy.

SJF sorts the waiting queue by ascending UNCACHED prefill work
`len(origin_input_ids) + len(output_ids) - num_matched_prefix_tokens`, so the many
cheap (cached-prefix) requests drain first -> lower MEAN TTFT. Unlike `lpm`, it stays
cost-aware at any queue size (lpm reverts to FCFS once the queue exceeds 128). Lossless:
changes only service ORDER. This validates the ordering in isolation.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from unittest.mock import MagicMock

from sglang.srt.managers.schedule_policy import CacheAgnosticPolicy, SchedulePolicy


def _req(rid, prompt_len, matched=0, output_len=0):
    r = MagicMock()
    r.rid = rid
    r.origin_input_ids = list(range(prompt_len))
    r.output_ids = list(range(output_len))
    r.num_matched_prefix_tokens = matched
    return r


class TestSJFSchedule(unittest.TestCase):
    def _order(self, reqs):
        q = list(reqs)
        SchedulePolicy._sort_by_shortest_prefill(q)
        return [r.rid for r in q]

    def test_sjf_enum_registered(self):
        self.assertEqual(CacheAgnosticPolicy("sjf"), CacheAgnosticPolicy.SJF)

    def test_shortest_uncached_first(self):
        # uncached suffix = prompt+output - matched
        a = _req("a", prompt_len=1000, matched=0)  # uncached 1000
        b = _req("b", prompt_len=100, matched=0)  # uncached 100  (cheapest)
        c = _req("c", prompt_len=5000, matched=0)  # uncached 5000
        self.assertEqual(self._order([a, b, c]), ["b", "a", "c"])

    def test_cache_hit_makes_request_cheap(self):
        # a long prompt that is mostly cached is cheaper than a short uncached one
        cached = _req("cached", prompt_len=10000, matched=9950)  # uncached 50
        short = _req("short", prompt_len=200, matched=0)  # uncached 200
        self.assertEqual(self._order([short, cached]), ["cached", "short"])

    def test_multiturn_output_counts_toward_work(self):
        # output_ids (prior turns) add to the sequence length to prefill
        a = _req("a", prompt_len=500, matched=100, output_len=0)  # uncached 400
        b = _req("b", prompt_len=500, matched=100, output_len=300)  # uncached 700
        self.assertEqual(self._order([a, b]), ["a", "b"])

    def test_fully_cached_is_zero_and_first(self):
        full = _req("full", prompt_len=8000, matched=8000)  # uncached 0
        some = _req("some", prompt_len=8000, matched=1000)  # uncached 7000
        self.assertEqual(self._order([some, full]), ["full", "some"])

    def test_matched_exceeding_len_clamped_nonnegative(self):
        # defensive: matched > seq len must not produce a negative key
        weird = _req("weird", prompt_len=100, matched=100000)
        normal = _req("normal", prompt_len=50, matched=0)  # uncached 50
        # weird clamps to 0 -> evicted... served first (cheapest)
        self.assertEqual(self._order([normal, weird]), ["weird", "normal"])


if __name__ == "__main__":
    unittest.main()
