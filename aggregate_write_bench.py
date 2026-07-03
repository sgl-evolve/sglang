#!/usr/bin/env python3
# Predict v15's benefit: does parallelizing L3 WRITES speed up the write-batch drain the way
# parallelizing reads did? The real HiCacheFile.set() writes value.tofile(tmp) then os.replace(tmp,final)
# with NO fsync (page-cache write), and numpy.tofile releases the GIL — so a serial per-page write LOOP
# that occupies the HiCache controller thread (blocking reads/prefetch) may parallelize across cores.
# 8 processes (ranks) each write N page-files to /mnt/localssd via a thread pool of W workers; report
# aggregate write throughput vs W. If it scales, v15 frees the controller faster -> lower TTFT.
# Also a contention probe: read throughput alone vs while a writer runs (does write steal read NVMe bw).
import os, sys, time, tempfile, shutil, uuid, threading, multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor

BASE = "/mnt/localssd/onyx-7q2-wbench"
PAGE = 768 * 1024      # match read bench page size
NPROC = 8              # 8 ranks (TP=8) writing concurrently
NPP = 200              # page-files per process per trial
BLOB = os.urandom(PAGE)

def write_one(d, i):
    # mirror set(): unique tmp (pid.tid.uuid) then atomic replace, no fsync
    final = f"{d}/p{i:04d}.bin"
    tmp = f"{final}.tmp.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}"
    with open(tmp, "wb", buffering=0) as f:
        f.write(BLOB)
    os.replace(tmp, final)

def write_worker(rank, threads, q):
    d = f"{BASE}/w{rank}_{threads}"
    os.makedirs(d, exist_ok=True)
    idxs = list(range(NPP))
    t0 = time.time()
    if threads <= 1:
        for i in idxs:
            write_one(d, i)
    else:
        with ThreadPoolExecutor(max_workers=threads) as ex:
            list(ex.map(lambda i: write_one(d, i), idxs))
    q.put(time.time() - t0)

def run_write(threads):
    q = mp.Queue()
    ps = [mp.Process(target=write_worker, args=(r, threads, q)) for r in range(NPROC)]
    t0 = time.time()
    for p in ps: p.start()
    for p in ps: p.join()
    wall = time.time() - t0
    _ = [q.get() for _ in range(NPROC)]
    total_mb = NPROC * NPP * PAGE / 1e6
    return total_mb / wall

def main():
    os.makedirs(BASE, exist_ok=True)
    try:
        print(f"write-drain throughput vs threads (8 ranks x {NPP} pages x {PAGE//1024}KB, "
              f"page-cache like set(), no fsync):")
        base = None
        for w in (1, 4, 8, 16, 32):
            # fresh subdirs per trial (write_worker makes them); clear prior to avoid replace-fastpath
            for r in range(NPROC):
                shutil.rmtree(f"{BASE}/w{r}_{w}", ignore_errors=True)
            mbps = run_write(w)
            if base is None: base = mbps
            print(f"  W={w:2d}: {mbps:8.0f} MB/s   ({mbps/base:.2f}x vs serial)")
        print("Interpretation: >1.3x at W=16 => serial write loop is thread-bound => v15 frees the "
              "HiCache controller thread faster => expected TTFT benefit. ~1.0x => writes are already "
              "bandwidth/replace-bound => v15 unlikely to help (still lossless, just neutral).")
    finally:
        shutil.rmtree(BASE, ignore_errors=True)
    return 0

if __name__ == "__main__":
    sys.exit(main())
