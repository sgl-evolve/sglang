import os, time, ctypes
from concurrent.futures import ThreadPoolExecutor
D="/mnt/localssd/kv-heron-eb9-iobench"
os.makedirs(D, exist_ok=True)
PAGE=732*1024          # ~ per-rank KV page (64 tok * ~11.4KB)
N=3000                 # ~2.1 GB total
BUF=os.urandom(PAGE)
paths=[os.path.join(D,f"p{i}.bin") for i in range(N)]

t=time.time()
for p in paths:
    with open(p,"wb",buffering=0) as f: f.write(BUF)
wt=time.time()-t
print(f"[write 1-thread] {N} pages {N*PAGE/1e9:.2f} GB in {wt:.2f}s -> {N*PAGE/1e9/wt:.2f} GB/s, {wt/N*1e3:.2f} ms/page", flush=True)

libc=ctypes.CDLL("libc.so.6", use_errno=True)
def drop_cache_reads():
    for p in paths:
        fd=os.open(p, os.O_RDONLY)
        libc.posix_fadvise(fd, 0, 0, 4)  # POSIX_FADV_DONTNEED
        os.close(fd)
drop_cache_reads()

def rd(p):
    b=bytearray(PAGE)
    with open(p,"rb",buffering=0) as f:
        if f.readinto(memoryview(b))!=PAGE: raise IOError("short")
    return len(b)

t=time.time(); tot=0
for p in paths: tot+=rd(p)
rt1=time.time()-t
print(f"[read 1-thread] {tot/1e9:.2f} GB in {rt1:.2f}s -> {tot/1e9/rt1:.2f} GB/s, {rt1/N*1e3:.2f} ms/page", flush=True)

for W in (4,8,16,32):
    drop_cache_reads()
    t=time.time()
    with ThreadPoolExecutor(max_workers=W) as ex:
        list(ex.map(rd, paths))
    rt=time.time()-t
    print(f"[read {W:>2}-thread] {tot/1e9:.2f} GB in {rt:.2f}s -> {tot/1e9/rt:.2f} GB/s, {rt/N*1e3:.3f} ms/page  (speedup {rt1/rt:.2f}x)", flush=True)

def wr(p):
    with open(p,"wb",buffering=0) as f: f.write(BUF)
for W in (8,16):
    for p in paths[:1000]:
        try: os.remove(p)
        except: pass
    t=time.time()
    with ThreadPoolExecutor(max_workers=W) as ex:
        list(ex.map(wr, paths[:1000]))
    wtp=time.time()-t
    print(f"[write {W:>2}-thread] 1000 pages in {wtp:.2f}s -> {1000*PAGE/1e9/wtp:.2f} GB/s (1-thread was {N*PAGE/1e9/wt:.2f} GB/s)", flush=True)

import shutil; shutil.rmtree(D, ignore_errors=True)
print("done", flush=True)
