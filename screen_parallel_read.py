#!/usr/bin/env python3
"""Screen: does parallelizing HiCache file-backend page reads speed up L3 disk reads?
Mimics the controller read pattern on real /mnt/localssd: many ~768KB page files,
read serially vs via a thread pool. Drops page cache (fadvise DONTNEED) so reads
truly hit NVMe (in prod the 768GB host tier is full, so pages aren't cached)."""
import os, sys, time, tempfile, shutil
from concurrent.futures import ThreadPoolExecutor

SSD = f"/mnt/localssd/onyx-7q2-screen"
PAGE_BYTES = 768 * 1024          # ~ per-rank KV bytes for a 64-token page
N = 800                          # ~51K-token prefix worth of pages
THREADS = [1, 8, 16, 32, 64]

def drop(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)

def read_one(path):
    buf = bytearray(PAGE_BYTES)
    with open(path, "rb", buffering=0) as f:
        n = f.readinto(buf)
    return n

def main():
    os.makedirs(SSD, exist_ok=True)
    paths = [os.path.join(SSD, f"p{i:05d}.bin") for i in range(N)]
    blob = os.urandom(PAGE_BYTES)
    for p in paths:
        with open(p, "wb", buffering=0) as f:
            f.write(blob)
        drop(p)
    total_mb = N * PAGE_BYTES / 1e6
    print(f"wrote {N} files x {PAGE_BYTES/1024:.0f}KB = {total_mb:.0f} MB on {SSD}")

    base = None
    for t in THREADS:
        for p in paths:
            drop(p)                      # force disk hit
        t0 = time.time()
        if t == 1:
            for p in paths:
                read_one(p)
        else:
            with ThreadPoolExecutor(max_workers=t) as ex:
                list(ex.map(read_one, paths))
        dt = time.time() - t0
        mbps = total_mb / dt
        if base is None:
            base = dt
        print(f"threads={t:>3}  time={dt*1000:8.1f} ms  throughput={mbps:8.0f} MB/s  speedup={base/dt:5.2f}x")
    shutil.rmtree(SSD, ignore_errors=True)

if __name__ == "__main__":
    main()
