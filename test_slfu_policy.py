#!/usr/bin/env python3
"""Offline unit test for the size-aware-LFU (`slfu`) eviction strategy.

Loads evict_policy.py in ISOLATION (no sglang package __init__, no torch/CUDA) and
verifies the eviction ordering with duck-typed mock nodes. The eviction site pops the
SMALLEST priority first (heapq), so smaller priority == evicted first == least valuable.

Run: python3 test_slfu_policy.py   (login node is fine; no GPU needed)
"""
import importlib.util
import os

HERE = os.path.dirname(os.path.abspath(__file__))
EP = os.path.join(HERE, "python/sglang/srt/mem_cache/evict_policy.py")
spec = importlib.util.spec_from_file_location("evict_policy_isolated", EP)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class MockKey:
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n


class MockNode:
    def __init__(self, hit_count, num_tokens, last_access_time):
        self.hit_count = hit_count
        self.key = MockKey(num_tokens)
        self.last_access_time = last_access_time


def evict_order(strategy, nodes):
    """Return nodes sorted the way the eviction heap pops them (smallest priority first)."""
    return sorted(nodes, key=strategy.get_priority)


def main():
    s = mod.SLFUStrategy()
    fails = []

    # 1. Reused node ALWAYS protected over any not-yet-reused node, even a huge one.
    reused_small = MockNode(hit_count=1, num_tokens=10, last_access_time=100.0)
    unused_huge = MockNode(hit_count=0, num_tokens=100000, last_access_time=999.0)
    order = evict_order(s, [reused_small, unused_huge])
    if order[0] is not unused_huge:
        fails.append("1: huge unused node must be evicted before a reused small node")

    # 2. Cold-start: among not-yet-reused nodes, the LARGE fresh document is RETAINED
    #    over cheap one-shot small prefixes (the key motivation for slfu).
    fresh_doc = MockNode(hit_count=0, num_tokens=50000, last_access_time=200.0)
    oneshot_a = MockNode(hit_count=0, num_tokens=20, last_access_time=201.0)
    oneshot_b = MockNode(hit_count=0, num_tokens=30, last_access_time=202.0)
    order = evict_order(s, [fresh_doc, oneshot_a, oneshot_b])
    if order[-1] is not fresh_doc:
        fails.append("2: large fresh document must be the LAST evicted among unused nodes")
    if not (order[0] is oneshot_a and order[1] is oneshot_b):
        fails.append("2b: among equal reuse, smaller nodes evicted first (size ascending)")

    # 3. LRU tiebreak among equal (hit_count, num_tokens): older (smaller time) first.
    old = MockNode(hit_count=2, num_tokens=100, last_access_time=50.0)
    new = MockNode(hit_count=2, num_tokens=100, last_access_time=500.0)
    order = evict_order(s, [new, old])
    if order[0] is not old:
        fails.append("3: among equal reuse+size, older node evicted first (LRU tiebreak)")

    # 4. Full-order sanity: primary=reuse, secondary=size(retain large), tertiary=recency.
    nodes = [
        MockNode(0, 20, 10.0),   # tiny one-shot -> evicted first
        MockNode(0, 5000, 11.0), # large fresh doc -> kept over tiny one-shot
        MockNode(1, 5000, 12.0), # reused large -> kept over all unused
        MockNode(3, 20, 13.0),   # hot small -> most protected
    ]
    got = [id(n) for n in evict_order(s, nodes)]
    want = [id(nodes[0]), id(nodes[1]), id(nodes[2]), id(nodes[3])]
    if got != want:
        fails.append(f"4: full ordering wrong; got {got} want {want}")

    # 5. None-key guard (root-like node) does not crash and sorts as size 0.
    try:
        nk = MockNode(0, 0, 1.0)
        nk.key = None
        _ = s.get_priority(nk)
    except Exception as e:
        fails.append(f"5: None key must not crash get_priority: {e!r}")

    if fails:
        print("FAIL:")
        for f in fails:
            print("  -", f)
        raise SystemExit(1)
    print("PASS: slfu eviction ordering correct (reuse > size-retention > recency); None-key safe")


if __name__ == "__main__":
    main()
