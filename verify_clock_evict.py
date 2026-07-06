#!/usr/bin/env python3
"""Offline verification of kv-lynx-4d2 CLOCK second-chance Mamba eviction.

Faithfully replicates UnifiedLRUList (unified_radix_cache.py:136-266) and the
victim-selection loop of MambaComponent.drive_eviction (mamba_component.py) with
my edit, using a mock "evict" (remove node from LRU, count 1 freed). Proves:
  (1) MAXSKIP=0 == stock strict-LRU order (byte-identical default -> lossless),
  (2) hot nodes are protected (skipped) up to the bounded budget,
  (3) eviction ALWAYS frees the requested count (progress guaranteed),
  (4) all-hot / single-node / over-request terminate and free what they can.
"""

# ---- replicate UnifiedLRUList (single component, no locks/host) ----
class Node:
    _c = 0
    def __init__(self, hit_count=0):
        self.id = Node._c; Node._c += 1
        self.hit_count = hit_count
        self.lru_prev = None
        self.lru_next = None

class LRU:
    def __init__(self):
        self.head = Node(); self.tail = Node()
        self.head.lru_next = self.tail
        self.tail.lru_prev = self.head
        self.cache = {}
    def _add_after(self, prev, new):
        new.lru_prev = prev; new.lru_next = prev.lru_next
        prev.lru_next.lru_prev = new; prev.lru_next = new
    def _add(self, node): self._add_after(self.head, node)
    def _remove(self, node):
        node.lru_prev.lru_next = node.lru_next
        node.lru_next.lru_prev = node.lru_prev
        node.lru_prev = None; node.lru_next = None
    def insert_mru(self, node):
        assert node.id not in self.cache
        self.cache[node.id] = node; self._add(node)
    def remove_node(self, node):
        assert node.id in self.cache
        del self.cache[node.id]; self._remove(node)
    def reset_node_mru(self, node):
        assert node.id in self.cache
        self._remove(node); self._add(node)
    def in_list(self, node):
        return node is not None and node.id in self.cache
    def get_prev_no_lock(self, node, check_id=True):
        if check_id: assert node.id in self.cache
        x = node.lru_prev
        # no locks in this model
        if x == self.head: return None
        return x
    def get_lru_no_lock(self):
        return self.get_prev_no_lock(self.tail, check_id=False)

# ---- replicate the drive_eviction victim loop (my edited version) ----
def drive_eviction(lru, request, maxskip, thr):
    freed = 0
    evicted_order = []
    x = lru.get_lru_no_lock()
    skips_left = maxskip
    guard = 0
    while freed < request and x is not None and lru.in_list(x):
        guard += 1
        assert guard < 100000, "NON-TERMINATION"
        if skips_left > 0 and getattr(x, "hit_count", 0) >= thr:
            x_next = lru.get_prev_no_lock(x)
            lru.reset_node_mru(x)
            skips_left -= 1
            if x_next is None or not lru.in_list(x_next):
                x_next = lru.get_lru_no_lock()
            x = x_next
            continue
        # mock evict: remove x, free 1, advance to next-colder-toward-warm
        x_next = lru.get_prev_no_lock(x)
        evicted_order.append(x.id)
        lru.remove_node(x)
        freed += 1
        if x_next is None or not lru.in_list(x_next):
            x_next = lru.get_lru_no_lock()
        x = x_next
    return freed, evicted_order

def build(hit_counts):
    """Insert nodes so index 0 is LRU (coldest), last is MRU."""
    Node._c = 0
    lru = LRU(); nodes = []
    for hc in hit_counts:
        n = Node(hit_count=hc); nodes.append(n); lru.insert_mru(n)
    return lru, nodes

fails = 0
def check(name, cond):
    global fails
    print(("PASS " if cond else "FAIL ") + name)
    if not cond: fails += 1

# (1) MAXSKIP=0 == stock strict LRU: evicts coldest-first (ns[0],ns[1],ns[2])
lru, ns = build([5,5,5,5,5])           # all hot, but skip disabled
freed, order = drive_eviction(lru, 3, maxskip=0, thr=2)
check("maxskip=0 frees exactly 3", freed == 3)
check("maxskip=0 strict-LRU order coldest-first",
      order == [ns[0].id, ns[1].id, ns[2].id])

# (2) hot nodes protected: coldest (ns[0]) is hot -> must be skipped, a COLD node
#     (ns[1]) evicted instead; ns[0] stays cached.
lru, ns = build([9, 0, 0, 0, 0])       # ns[0] hot (coldest), rest cold
freed, order = drive_eviction(lru, 1, maxskip=4, thr=2)
check("hot-coldest protected: frees 1", freed == 1)
check("hot-coldest protected: evicts COLD ns[1], ns[0] survives",
      order == [ns[1].id] and ns[0].id in lru.cache)

# (3) always frees requested even with many hot nodes (progress guaranteed)
lru, ns = build([9,9,9,0,0,9,9])
freed, order = drive_eviction(lru, 4, maxskip=3, thr=2)
check("mixed frees exactly 4", freed == 4)

# (4a) ALL hot, skip budget < request: must still free request (strict-LRU fallback)
lru, ns = build([7,7,7,7,7,7])
freed, order = drive_eviction(lru, 4, maxskip=2, thr=2)
check("all-hot frees exactly 4 (fallback)", freed == 4)

# (4b) all hot, huge skip budget, request all: still terminates + frees all
lru, ns = build([7,7,7])
freed, order = drive_eviction(lru, 3, maxskip=1000, thr=2)
check("all-hot request-all terminates+frees 3", freed == 3)

# (4c) single node
lru, ns = build([9])
freed, order = drive_eviction(lru, 1, maxskip=5, thr=2)
check("single hot node frees 1", freed == 1)

# (4d) over-request: request more than exist -> frees all, terminates
lru, ns = build([0,9,0])
freed, order = drive_eviction(lru, 10, maxskip=5, thr=2)
check("over-request frees all 3 + terminates", freed == 3)

# (5) determinism: MAXSKIP=0 identical across two runs, coldest-first order
lru1,n1 = build([3,1,4,1,5,9,2,6]); f1,o1 = drive_eviction(lru1,5,0,2)
lru2,n2 = build([3,1,4,1,5,9,2,6]); f2,o2 = drive_eviction(lru2,5,0,2)
expect = [n1[i].id for i in range(5)]   # coldest-first = insertion order
check("maxskip=0 deterministic + coldest-first order", o1==o2==expect)

print("\n" + ("ALL PASS" if fails==0 else f"{fails} FAILED"))
