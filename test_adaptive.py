#!/usr/bin/env python3
"""Logic test for the congestion-aware 'adaptive' prefetch stop policy.
Verifies: idle disk -> wait ~full linear-timeout (don't give up early);
congested disk (deep backlog) -> give up early (terminate -> recompute).
Tests the pure method against a mock self (no full class instantiation)."""
import time, types
from queue import Queue

from sglang.srt.mem_cache.hi_mamba_radix_cache import (
    HiMambaRadixCache,
    _ADAPTIVE_CONGESTION_ALPHA,
    _ADAPTIVE_MIN_WAIT,
)
from sglang.srt.mem_cache.hicache_storage import PrefetchTimeoutConfig

fn = HiMambaRadixCache._adaptive_should_terminate

def make_cc(backlog):
    buf = Queue(); pq = Queue()
    for _ in range(backlog):
        buf.put(object())
    return types.SimpleNamespace(prefetch_buffer=buf, prefetch_queue=pq)

def make_op(npages, age_s):
    return types.SimpleNamespace(
        hash_value=["h"] * npages,
        start_time=time.monotonic() - age_s,
    )

PAGE = 64
cfg = PrefetchTimeoutConfig(base=2.0, per_ki_token=0.1, max=30.0)

# case A: idle disk (backlog 0), small age -> must NOT terminate (waits for read)
selfA = types.SimpleNamespace(cache_controller=make_cc(0),
                              prefetch_timeout_config=cfg, page_size=PAGE)
opA = make_op(npages=16, age_s=0.5)          # 1024 tok -> base~2.1s; age 0.5s < 2.1s
assert fn(selfA, opA) is False, "idle disk should keep waiting"

# case B: idle disk but aged past the base timeout -> terminate (timeout-like)
opB = make_op(npages=16, age_s=5.0)          # age 5s > ~2.1s
assert fn(selfA, opB) is True, "idle disk past timeout should terminate"

# case C: congested disk (deep backlog) shrinks the budget -> terminate early
selfC = types.SimpleNamespace(cache_controller=make_cc(50),
                              prefetch_timeout_config=cfg, page_size=PAGE)
opC = make_op(npages=16, age_s=0.5)          # same 0.5s age that waited in case A
# eff = max(0.2, 2.1/(1+alpha*50)) ~ 0.2s  -> 0.5s > 0.2s -> terminate
assert fn(selfC, opC) is True, "congested disk should give up early (recompute)"

# case D: congestion never drops below MIN_WAIT floor
selfD = types.SimpleNamespace(cache_controller=make_cc(10000),
                              prefetch_timeout_config=cfg, page_size=PAGE)
opD = make_op(npages=16, age_s=0.1)          # age 0.1s < MIN_WAIT 0.2s -> keep waiting
assert fn(selfD, opD) is False, "must respect MIN_WAIT floor even under huge backlog"

print(f"PASS adaptive: idle-waits / timeout-terminates / congested-gives-up-early / "
      f"MIN_WAIT floor respected (alpha={_ADAPTIVE_CONGESTION_ALPHA}, min={_ADAPTIVE_MIN_WAIT}s)")
