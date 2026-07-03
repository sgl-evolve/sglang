#!/usr/bin/env python3
"""Decisive screen: does per-rank read parallelism unlock aggregate NVMe bandwidth
BEYOND what 8 TP ranks reading serially already get? Simulates 8 ranks = 8 processes
reading their own cold shards from the shared /mnt/localssd md array concurrently.
W=1 per process == baseline (each rank's single serial IO thread). W>1 == kv-lynx-4d2
parallel IO. If aggregate(W=1) already ~= aggregate(W=8), the array is saturated and
parallel IO is a dud; if aggregate scales with W, it's a real win."""
import os, sys, time, shutil, multiprocessing as mp

PAGE = 786432
NPROC = 8                 # 8 TP ranks
NFILES_PER = 1200         # ~0.9 GB per proc, ~7.2 GB total (exceeds any warm cache slice)
BASE = "/mnt/localssd/kv-lynx-4d2-mpbench"

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
        return f.readinto(buf)

def worker(rank, workers, barrier, q):
    d = os.path.join(BASE, f"r{rank}")
    paths = [os.path.join(d, f"p{i}.bin") for i in range(NFILES_PER)]
    evict(paths)
    barrier.wait()                       # all ranks start together
    t = time.monotonic()
    total = 0
    if workers <= 1:
        for p in paths:
            total += read_one(p)
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for fut in as_completed([ex.submit(read_one, p) for p in paths]):
                total += fut.result()
    dt = time.monotonic() - t
    q.put((rank, total, dt))

def setup():
    blob = os.urandom(PAGE)
    for r in range(NPROC):
        d = os.path.join(BASE, f"r{r}")
        os.makedirs(d, exist_ok=True)
        for i in range(NFILES_PER):
            with open(os.path.join(d, f"p{i}.bin"), "wb", buffering=0) as f:
                f.write(blob)
    os.sync()

def run(workers):
    barrier = mp.Barrier(NPROC)
    q = mp.Queue()
    procs = [mp.Process(target=worker, args=(r, workers, barrier, q)) for r in range(NPROC)]
    t0 = time.monotonic()
    for p in procs: p.start()
    res = [q.get() for _ in range(NPROC)]
    for p in procs: p.join()
    wall = time.monotonic() - t0
    total_bytes = sum(b for _, b, _ in res)
    # aggregate over the max per-rank read window (ranks start together via barrier)
    max_dt = max(dt for _, _, dt in res)
    return total_bytes / max_dt / 1e9, total_bytes / 1e9, max_dt

def main():
    os.makedirs(BASE, exist_ok=True)
    try:
        print(f"setup: {NPROC} procs x {NFILES_PER} x {PAGE/1024:.0f}KB = "
              f"{NPROC*NFILES_PER*PAGE/1e9:.2f} GB", flush=True)
        setup()
        print("=== aggregate cold-read bandwidth: 8 ranks concurrent, vs workers/rank ===", flush=True)
        print("  (workers=1 == baseline serial per-rank IO thread)", flush=True)
        for w in [1, 2, 4, 8, 16]:
            gbps, gb, dt = run(w)
            print(f"  workers/rank={w:<3d}  AGGREGATE {gbps:6.2f} GB/s   ({gb:.2f} GB in {dt:.2f}s)", flush=True)
    finally:
        shutil.rmtree(BASE, ignore_errors=True)

if __name__ == "__main__":
    mp.set_start_method("fork")
    main()
