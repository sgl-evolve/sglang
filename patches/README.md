# exclusive_tiering.patch — the isolated, reviewable mechanism

The complete engine change for **lossless exclusive L1↔L2 HiCache tiering** (`SGLANG_HICACHE_EXCLUSIVE`),
extracted clean of all research tooling / report / sim commits. Two files only:

- `python/sglang/srt/environ.py` (+11): `SGLANG_HICACHE_EXCLUSIVE`, `SGLANG_HICACHE_EXCLUSIVE_HOT_KEEP`.
- `python/sglang/srt/mem_cache/unified_radix_cache.py` (+80/−12): `_promote_free_host` + three gated edits
  (`_inc_hit_count` early-return, evict-time D→H backup in `_evict_device_leaf`, consensus `writing_check` /
  `sanity_check`), all reusing the mature `write_back` plumbing.

Base commit: `21023af6d^` = `a334877e5` (verified `git apply --check` clean onto that base).

```bash
git apply patches/exclusive_tiering.patch      # or: git am / patch -p1
```

Default OFF (`EnvBool(False)`) — zero behavior change unless `SGLANG_HICACHE_EXCLUSIVE=1`. Rationale,
evidence (hit 0.62→0.75, +13pp, bit-exact lossless), correctness argument, and the two-sided generalization
(capacity band boundary + transfer-cost scope caveat) are in `../UPSTREAM.md` and `../report.md`.
