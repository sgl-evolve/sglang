#!/usr/bin/env python3
"""Losslessness test for kv-lynx-4d2 parallel HiCacheFile IO (CPU-only, login node).
Round-trips random KV pages through the generic batch_set/batch_get path with
serial (workers=1) vs parallel (workers=16) backends and asserts byte-identical
results and preserved ordering. This is the dominant KV read/write path."""
import os, tempfile, torch

tmp = tempfile.mkdtemp(prefix="kvlynx_iotest_")
os.environ["SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR"] = tmp

from sglang.srt.mem_cache.hicache_storage import HiCacheFile, HiCacheStorageConfig

def make_backend(workers, sub):
    d = os.path.join(tmp, sub)
    os.makedirs(d, exist_ok=True)
    os.environ["SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR"] = d
    cfg = HiCacheStorageConfig(
        tp_rank=0, tp_size=1, pp_rank=0, pp_size=1,
        attn_cp_rank=0, attn_cp_size=1, is_mla_model=False,
        enable_storage_metrics=False, is_page_first_layout=True,
        model_name="test-model", extra_config={"hicache_io_workers": workers},
    )
    return HiCacheFile(cfg)

N, PAGE_ELEMS = 200, 4096
torch.manual_seed(0)
values = [torch.randn(PAGE_ELEMS, dtype=torch.float32) for _ in range(N)]
keys = [f"key_{i:04d}" for i in range(N)]

def roundtrip(workers, sub):
    be = make_backend(workers, sub)
    ok = be.batch_set(keys, values)
    assert ok, f"batch_set failed (workers={workers})"
    targets = [torch.empty(PAGE_ELEMS, dtype=torch.float32) for _ in range(N)]
    got = be.batch_get(keys, targets)
    assert all(g is not None for g in got), f"batch_get had misses (workers={workers})"
    return targets

serial = roundtrip(1, "serial")
par = roundtrip(16, "parallel")

# 1) parallel round-trip is byte-identical to the original values (lossless)
for i in range(N):
    assert torch.equal(par[i], values[i]), f"parallel mismatch at page {i}"
# 2) parallel == serial (ordering + content preserved)
for i in range(N):
    assert torch.equal(par[i], serial[i]), f"parallel != serial at page {i}"

print(f"PASS: {N} pages round-tripped byte-identical; serial==parallel; ordering preserved")
import shutil; shutil.rmtree(tmp, ignore_errors=True)
