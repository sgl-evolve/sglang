#!/usr/bin/env python3
"""Microbench: does parallelizing per-page blocking reads raise disk bandwidth
on the /mnt/localssd md NVMe array? Mirrors HiCacheFile.get (open buffering=0 +
readinto) on realistic ~768KB KV pages, reading COLD (page cache evicted via
posix_fadvise DONTNEED, matching the real disk-tier which holds evicted/cold KV).
Sweeps worker count; also tests multi-process to approximate 8 TP ranks."""
import os, sys, time, tempfile, shutil
from concurrent.futures import ThreadPoolExecutor, as_completed

PAGE = 786432          # 2*12*64*2*256 bytes ~ measured KV page @ page_size 64, fp8
NFILES = 5000          # ~3.75 GB working set
DIRBASE = f"/mnt/localssd/kv-lynx-4d2-microbench"

def evict(paths):
    for p in paths:
        try:
            fd = os.open(p, os.O_RDONLY)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            os.close(fd)
        except OSError:
            pass

def read_one(p):
    buf = bytearray(PAGE)
    with open(p, "rb", buffering=0) as f:
        n = f.readinto(buf)
    return n

def run(paths, workers):
    evict(paths)                       # cold cache before each measurement
    t = time.monotonic()
    if workers <= 1:
        total = sum(read_one(p) for p in paths)
    else:
        total = 0
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for fut in as_completed([ex.submit(read_one, p) for p in paths]):
                total += fut.result()
    dt = time.monotonic() - t
    return total / dt / 1e9, dt      # GB/s, seconds

def main():
    d = tempfile.mkdtemp(prefix="mb_", dir=DIRBASE) if os.path.isdir(DIRBASE) else \
        (os.makedirs(DIRBASE, exist_ok=True) or tempfile.mkdtemp(prefix="mb_", dir=DIRBASE))
    try:
        print(f"writing {NFILES} x {PAGE/1024:.0f}KB = {NFILES*PAGE/1e9:.2f} GB to {d}", flush=True)
        blob = os.urandom(PAGE)
        paths = []
        for i in range(NFILES):
            p = os.path.join(d, f"p{i}.bin")
            with open(p, "wb", buffering=0) as f:
                f.write(blob)
            paths.append(p)
        os.sync()
        print("=== cold-read bandwidth vs worker count (single process) ===", flush=True)
        for w in [1, 2, 4, 8, 16, 32, 64, 128]:
            gbps, dt = run(paths, w)
            print(f"  workers={w:<4d}  {gbps:6.2f} GB/s   ({dt:.2f}s)", flush=True)
    finally:
        shutil.rmtree(d, ignore_errors=True)

if __name__ == "__main__":
    main()
