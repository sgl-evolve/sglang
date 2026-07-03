#!/usr/bin/env python3
"""Screen: is the per-hit-query os.scandir over the whole L3 dir a bottleneck?
_collect_existing_component_keys scans ALL files in the storage dir and filters to
a ~256-name target set, on EVERY prefetch hit-query, on every rank. Compare that to
the O(keys) direct os.path.exists fix, at realistic file counts."""
import os, time, tempfile, shutil

BASE = "/mnt/localssd/kv-lynx-4d2-scandir"
TARGET_KEYS = 256          # keys(128) x (KV + 1 mamba pool)

def scandir_filter(d, target_set):
    existing = set()
    with os.scandir(d) as it:
        for e in it:
            if e.is_file() and e.name in target_set:
                existing.add(e.name)
    return existing

def direct_exists(d, target_names):
    return {n for n in target_names if os.path.exists(os.path.join(d, n))}

def main():
    os.makedirs(BASE, exist_ok=True)
    d = tempfile.mkdtemp(prefix="sd_", dir=BASE)
    try:
        for nfiles in [200_000, 500_000, 1_000_000]:
            # top up to nfiles
            have = sum(1 for _ in os.scandir(d))
            print(f"creating up to {nfiles} files (have {have})...", flush=True)
            for i in range(have, nfiles):
                open(os.path.join(d, f"h{i:08d}.bin"), "wb").close()
            # target set: 256 real existing files (present -> worst case for scandir filter)
            names = [f"h{i:08d}.bin" for i in range(0, nfiles, max(1, nfiles // TARGET_KEYS))][:TARGET_KEYS]
            tset = set(names)
            # warm dir cache
            scandir_filter(d, tset)
            t = time.monotonic(); [scandir_filter(d, tset) for _ in range(3)]
            sd = (time.monotonic() - t) / 3
            t = time.monotonic(); [direct_exists(d, names) for _ in range(3)]
            dx = (time.monotonic() - t) / 3
            print(f"  files={nfiles:>9d}  scandir_filter={sd*1000:8.1f} ms   "
                  f"direct_exists={dx*1000:6.2f} ms   speedup={sd/dx:6.0f}x", flush=True)
    finally:
        shutil.rmtree(d, ignore_errors=True)

if __name__ == "__main__":
    main()
