#!/usr/bin/env python3
"""Cheap screen: serial vs parallel small-file reads on /mnt/localssd (the HiCache L3 tier).
Mimics HiCacheFile.get(): open(buffering=0).readinto() per page-file. Drops page cache via
posix_fadvise(DONTNEED) so we measure real NVMe reads, not RAM. No model, ~2 min."""
import os, sys, time, shutil
from concurrent.futures import ThreadPoolExecutor

BASE = "/mnt/localssd/kv-heron-e29-bench"
SIZES = [64*1024, 256*1024, 1024*1024]   # per-page-file bytes (real KV page ~256KB-1MB per rank)
THREADS = [1, 4, 8, 16, 32, 64]
TOTAL_BYTES = int(1.5 * 1024**3)         # ~1.5 GB per size

def drop(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)

def read_one(path, size):
    buf = bytearray(size)
    with open(path, "rb", buffering=0) as f:
        n = f.readinto(buf)
    drop(path)
    return n

def run():
    os.makedirs(BASE, exist_ok=True)
    print(f"{'size_KB':>8} {'nfiles':>7} {'threads':>7} {'secs':>7} {'MB/s':>9} {'pages/s':>9} {'speedup':>8}", flush=True)
    for size in SIZES:
        nfiles = max(64, TOTAL_BYTES // size)
        d = os.path.join(BASE, f"s{size}")
        shutil.rmtree(d, ignore_errors=True); os.makedirs(d)
        paths = [os.path.join(d, f"p{i}.bin") for i in range(nfiles)]
        chunk = os.urandom(size)
        for p in paths:
            with open(p, "wb", buffering=0) as f:
                f.write(chunk)
            drop(p)
        base = None
        for nt in THREADS:
            for p in paths: drop(p)
            t0 = time.time()
            if nt == 1:
                for p in paths: read_one(p, size)
            else:
                with ThreadPoolExecutor(max_workers=nt) as ex:
                    list(ex.map(lambda p: read_one(p, size), paths))
            secs = time.time() - t0
            if nt == 1: base = secs
            print(f"{size//1024:>8} {nfiles:>7} {nt:>7} {secs:>7.2f} "
                  f"{nfiles*size/secs/1e6:>9.1f} {nfiles/secs:>9.0f} {base/secs:>7.2f}x", flush=True)
        shutil.rmtree(d, ignore_errors=True)
    shutil.rmtree(BASE, ignore_errors=True)

if __name__ == "__main__":
    run()
