#!/usr/bin/env python3
# Predict optimal SGLANG_HICACHE_FILE_READ_THREADS under real 8-rank concurrency:
# 8 processes (ranks) each read N page-files from /mnt/localssd via a thread pool; report
# aggregate read throughput vs per-process thread count. Page-cache dropped (fadvise) so reads hit NVMe.
import os,sys,time,tempfile,shutil,multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
BASE="/mnt/localssd/onyx-7q2-iobench"; PAGE=768*1024; NPROC=8; NPP=200  # per-process files
def drop(p):
    fd=os.open(p,os.O_RDONLY)
    try: os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED)
    finally: os.close(fd)
def rd(p):
    b=bytearray(PAGE)
    with open(p,'rb',buffering=0) as f: f.readinto(b)
def worker(rank,threads,q):
    d=f"{BASE}/r{rank}"; paths=[f"{d}/p{i:04d}.bin" for i in range(NPP)]
    for p in paths: drop(p)
    t0=time.time()
    with ThreadPoolExecutor(max_workers=threads) as ex: list(ex.map(rd,paths))
    q.put(time.time()-t0)
def setup():
    blob=os.urandom(PAGE)
    for r in range(NPROC):
        d=f"{BASE}/r{r}"; os.makedirs(d,exist_ok=True)
        for i in range(NPP):
            with open(f"{d}/p{i:04d}.bin",'wb',buffering=0) as f: f.write(blob)
def run(threads):
    q=mp.Queue(); ps=[mp.Process(target=worker,args=(r,threads,q)) for r in range(NPROC)]
    t0=time.time()
    for p in ps:p.start()
    for p in ps:p.join()
    wall=time.time()-t0
    total_mb=NPROC*NPP*PAGE/1e6
    return total_mb/wall
if __name__=="__main__":
    os.makedirs(BASE,exist_ok=True); setup()
    total_mb=NPROC*NPP*PAGE/1e6
    print(f"8 procs x {NPP} files x {PAGE//1024}KB = {total_mb:.0f} MB aggregate on {BASE}")
    for th in [4,8,16,32,64]:
        mbps=run(th)
        print(f"per-proc-threads={th:>3} (aggregate {th*NPROC:>3})  aggregate throughput={mbps:8.0f} MB/s")
    shutil.rmtree(BASE,ignore_errors=True)
