#!/usr/bin/env python3
"""Faithful microbench for the HiCacheFile L3 page IO: does parallelizing the serial
per-page loop help on /mnt/localssd? Mimics _read_page (fresh pinned dummy + readinto)
and _write_page (pinned buffer .tofile + os.replace). Compares serial vs ThreadPool,
and a reuse-pinned-buffer-per-thread read variant (to isolate cudaHostAlloc cost).

Usage: mbench_file_io.py <scratch_dir> [page_bytes_csv] [num_pages] [threads]
"""
import os, sys, time, threading, uuid
from concurrent.futures import ThreadPoolExecutor
import torch


def human(bps):
    for u in ["B/s", "KB/s", "MB/s", "GB/s"]:
        if bps < 1024:
            return f"{bps:.1f} {u}"
        bps /= 1024
    return f"{bps:.1f} TB/s"


def write_page(path, buf):
    tmp = f"{path}.tmp.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}"
    buf.view(torch.uint8).numpy().tofile(tmp)
    os.replace(tmp, path)
    return True


def read_page_fresh(path, nbytes):
    dummy = torch.zeros(nbytes, dtype=torch.uint8, pin_memory=True)  # fresh pinned alloc
    with open(path, "rb", buffering=0) as f:
        mv = memoryview(dummy.numpy())
        return f.readinto(mv) == nbytes


_tls = threading.local()
def read_page_reuse(path, nbytes):
    buf = getattr(_tls, "buf", None)
    if buf is None or buf.numel() != nbytes:
        buf = torch.zeros(nbytes, dtype=torch.uint8, pin_memory=True)
        _tls.buf = buf
    with open(path, "rb", buffering=0) as f:
        mv = memoryview(buf.numpy())
        return f.readinto(mv) == nbytes


def bench(scratch, page_bytes, num_pages, threads):
    d = os.path.join(scratch, f"mb_{page_bytes}")
    os.makedirs(d, exist_ok=True)
    for fn in os.listdir(d):
        os.remove(os.path.join(d, fn))
    paths = [os.path.join(d, f"p{i}.bin") for i in range(num_pages)]
    src = torch.zeros(page_bytes, dtype=torch.uint8, pin_memory=True)
    total = page_bytes * num_pages

    def drop_caches_note():
        # We cannot drop page cache without root; reads after fresh write may be cache-hot.
        # To approximate cold reads we re-write fresh files each read-round below is overkill;
        # instead we report both, knowing serial vs parallel are compared under identical cache state.
        pass

    # ---- WRITE serial
    for fn in os.listdir(d):
        os.remove(os.path.join(d, fn))
    t = time.perf_counter()
    for p in paths:
        write_page(p, src)
    w_ser = time.perf_counter() - t

    # ---- WRITE threaded
    for fn in os.listdir(d):
        os.remove(os.path.join(d, fn))
    t = time.perf_counter()
    with ThreadPoolExecutor(max_workers=threads) as ex:
        list(ex.map(lambda p: write_page(p, src), paths))
    w_thr = time.perf_counter() - t

    # ensure files exist for reads (threaded write left them)
    # ---- READ serial (fresh pinned)
    t = time.perf_counter()
    for p in paths:
        read_page_fresh(p, page_bytes)
    r_ser = time.perf_counter() - t

    # ---- READ threaded (fresh pinned)
    t = time.perf_counter()
    with ThreadPoolExecutor(max_workers=threads) as ex:
        list(ex.map(lambda p: read_page_fresh(p, page_bytes), paths))
    r_thr_fresh = time.perf_counter() - t

    # ---- READ threaded (reuse pinned per thread)
    t = time.perf_counter()
    with ThreadPoolExecutor(max_workers=threads) as ex:
        list(ex.map(lambda p: read_page_reuse(p, page_bytes), paths))
    r_thr_reuse = time.perf_counter() - t

    for fn in os.listdir(d):
        os.remove(os.path.join(d, fn))
    os.rmdir(d)

    print(f"\n== page={page_bytes/1024:.0f}KB x {num_pages} pages ({total/1024/1024:.0f} MB), threads={threads} ==")
    print(f"  WRITE serial   {w_ser*1e3:8.1f} ms  {human(total/w_ser)}")
    print(f"  WRITE threaded {w_thr*1e3:8.1f} ms  {human(total/w_thr)}   speedup x{w_ser/w_thr:.2f}")
    print(f"  READ  serial   {r_ser*1e3:8.1f} ms  {human(total/r_ser)}")
    print(f"  READ  thr/fresh{r_thr_fresh*1e3:8.1f} ms  {human(total/r_thr_fresh)}   speedup x{r_ser/r_thr_fresh:.2f}")
    print(f"  READ  thr/reuse{r_thr_reuse*1e3:8.1f} ms  {human(total/r_thr_reuse)}   speedup x{r_ser/r_thr_reuse:.2f}")


def main():
    scratch = sys.argv[1] if len(sys.argv) > 1 else "/mnt/localssd/kv-onyx-mf76-mbench"
    sizes = [int(x) for x in (sys.argv[2].split(",") if len(sys.argv) > 2 else ["262144", "1048576", "4194304"])]
    num_pages = int(sys.argv[3]) if len(sys.argv) > 3 else 256
    threads = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    os.makedirs(scratch, exist_ok=True)
    print(f"scratch={scratch} num_pages={num_pages} threads={threads}")
    for s in sizes:
        bench(scratch, s, num_pages, threads)


if __name__ == "__main__":
    main()
