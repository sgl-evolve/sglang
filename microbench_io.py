"""Micro-benchmark: serial vs thread-pool per-page file I/O on local NVMe.

Mimics HiCacheFile.set (arr.tofile(tmp)+os.replace) and .get
(open(buffering=0)+readinto), which are the exact hot-path calls. Reads are
made COLD via posix_fadvise(DONTNEED) so we measure the SSD, not page cache.
This validates the v1 mechanism (parallel L3 I/O) cheaply, without a model load.
"""
import os, sys, time, uuid, threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np

DIR = sys.argv[1] if len(sys.argv) > 1 else "/mnt/localssd/quill-7m3/mb"
PAGE_BYTES = int(sys.argv[2]) if len(sys.argv) > 2 else 768 * 1024
N = int(sys.argv[3]) if len(sys.argv) > 3 else 1500
WORKER_SWEEP = [1, 4, 8, 16, 32]

os.makedirs(DIR, exist_ok=True)
blob = np.frombuffer(bytes(range(256)) * (PAGE_BYTES // 256 + 1), dtype=np.uint8)[:PAGE_BYTES].copy()
paths = [os.path.join(DIR, f"pg_{i}.bin") for i in range(N)]

def write_one(p):
    tmp = f"{p}.tmp.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}"
    blob.tofile(tmp)
    os.replace(tmp, p)
    return True

def cold(p):
    try:
        fd = os.open(p, os.O_RDONLY)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(fd)
    except OSError:
        pass

def read_one(p):
    buf = bytearray(PAGE_BYTES)
    with open(p, "rb", buffering=0) as f:
        n = f.readinto(memoryview(buf))
    return n

def timed(fn, items, workers):
    if workers <= 1:
        t0 = time.monotonic()
        for it in items:
            fn(it)
        return time.monotonic() - t0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        t0 = time.monotonic()
        list(ex.map(fn, items))
        return time.monotonic() - t0

total_gb = N * PAGE_BYTES / 1e9
print(f"dir={DIR} page={PAGE_BYTES/1024:.0f}KB n={N} total={total_gb:.2f}GB")
print(f"{'op':>6} {'workers':>8} {'sec':>8} {'GB/s':>8} {'kIOPS':>8} {'speedup':>8}")

base = {}
for op, fn, needs_cold in [("write", write_one, False), ("read", read_one, True)]:
    for w in WORKER_SWEEP:
        if needs_cold:
            # ensure files exist (from write phase) and evict from cache
            for p in paths:
                cold(p)
        else:
            for p in paths:
                try:
                    os.remove(p)
                except OSError:
                    pass
        dt = timed(fn, paths, w)
        gbps = total_gb / dt
        kiops = N / dt / 1000
        if w == 1:
            base[op] = dt
        sp = base[op] / dt
        print(f"{op:>6} {w:>8} {dt:>8.3f} {gbps:>8.2f} {kiops:>8.1f} {sp:>7.2f}x")

# cleanup
for p in paths:
    try:
        os.remove(p)
    except OSError:
        pass
print("done")
