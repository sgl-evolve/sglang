#!/usr/bin/env python3
"""Losslessness test for the _collect_existing_component_keys fix: the new direct
os.path.isfile probe must return EXACTLY the same set as the original whole-dir
os.scandir filter, for arbitrary present/absent KV + pool-component files."""
import os, tempfile
from dataclasses import dataclass

tmp = tempfile.mkdtemp(prefix="kvlynx_sdfix_")
os.environ["SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR"] = tmp
from sglang.srt.mem_cache.hicache_storage import (
    HiCacheFile,
    HiCacheStorageConfig,
    PoolHitPolicy,
)

@dataclass
class FakeTransfer:
    name: str
    keys: list = None
    hit_policy: PoolHitPolicy = PoolHitPolicy.ALL_PAGES

def ref_scandir(be, keys, pool_transfers):
    """Original implementation (whole-dir scandir + filter)."""
    target = {f"{be._get_component_key(k)}.bin" for k in keys}
    for t in pool_transfers or []:
        for k in keys:
            target.add(f"{be._get_component_key(k, t.name)}.bin")
    existing = set()
    with os.scandir(be.file_path) as it:
        for e in it:
            if e.is_file() and e.name in target:
                existing.add(e.name)
    return existing

cfg = HiCacheStorageConfig(
    tp_rank=0, tp_size=1, pp_rank=0, pp_size=1, attn_cp_rank=0, attn_cp_size=1,
    is_mla_model=False, enable_storage_metrics=False, is_page_first_layout=True,
    model_name="test-model", extra_config={},
)
be = HiCacheFile(cfg)
keys = [f"k{i:03d}" for i in range(40)]
pool = [FakeTransfer("mamba")]

# create a realistic mix: KV present for a contiguous prefix + some gaps; mamba
# present for a subset; plus unrelated noise files that must be ignored.
import random
random.seed(1)
present_kv = set(range(25)) - {17}          # contiguous-ish with a gap at 17
present_mamba = {i for i in range(40) if random.random() < 0.6}
for i in present_kv:
    open(os.path.join(tmp, f"{be._get_component_key(keys[i])}.bin"), "wb").close()
for i in present_mamba:
    open(os.path.join(tmp, f"{be._get_component_key(keys[i], 'mamba')}.bin"), "wb").close()
for j in range(200):                          # noise
    open(os.path.join(tmp, f"noise_{j}.bin"), "wb").close()

got = be._collect_existing_component_keys(keys, pool)
ref = ref_scandir(be, keys, pool)
assert got == ref, f"MISMATCH\n only_new={got-ref}\n only_ref={ref-got}"

# also exercise batch_exists_v2 end-to-end (KV prefix capped by the gap at 17)
pool_full = [FakeTransfer("mamba", keys=keys)]
res = be.batch_exists_v2(keys, pool_full)
assert 0 <= res.kv_hit_pages <= 17, f"kv prefix must be <=17 (gap at 17), got {res.kv_hit_pages}"

print(f"PASS: fix returns identical set as scandir ({len(got)} files); "
      f"batch_exists_v2 kv prefix={res.kv_hit_pages} (<=17 ok)")
import shutil; shutil.rmtree(tmp, ignore_errors=True)
