# v0 baseline live profile (the real bottlenecks)

Captured live from `/metrics` during the v0 run (per TP rank 0 unless noted).

## ShareGPT (bench_mix, 60 clients) — light tiering
- num_running ~55, **num_queue ~0** (NOT queue-bound; system keeps up)
- device token_usage ~0.61
- **host L2 only ~9.6% full** (399K / 4.17M tokens) → **disk L3 essentially unused** (no prefetch)
- load_back ~393K, backup ~395K, cache_hit_rate ~0.42
- => ShareGPT bottleneck is NOT KV tiering; disk-IO optimizations won't help it.

## LooGLE (bench_serving, rate=10, multiturn ~17K-token contexts) — tiering-stressed
- **num_queue ~111**, num_running ~20 → **heavily QUEUE-BOUND** (rate>throughput on long contexts)
- device token_usage ~0.70
- **host L2 ~99.98% FULL** (4.166M / 4.167M) → constant eviction
- **load_back ~23.4M tokens (L2→L1) — DOMINANT data movement** (~91K/req avg ≫ 17K context
  => prefixes evicted between turns and reloaded = THRASHING; device cache_hit_rate ≈ 0)
- disk **backup ~5.0M** (write_through) + **prefetch ~0.90M** (L3→L2) → disk L3 IS active (single
  serial backup/prefetch thread + serial `HiCacheFile.batch_set/batch_get`)
- disk tier ~677 GB and growing (capped at 2 TB), correctly on /mnt/localssd

## Implications for v-search (mean TTFT headline)
1. LooGLE dominates the headline (long contexts, deep queue). Its TTFT = queue_wait + load_back +
   chunked prefill.
2. Biggest data movement = **load_back (L2→L1, 23M)**, already layer-wise pipelined (hard to beat).
3. Disk L3 IS on the path (backup 5M gates host turnover; prefetch 0.9M gates load-from-disk).
   `batch_set`/`batch_get` are serial single-thread → parallelizing is lossless + low-risk (v1).
4. Thrashing (hit_rate~0, load_back≫context) points to scheduling/admission (cache-aware ordering,
   keep-resident) as a larger but riskier lever (v2+).

## v1 decision
v1 = **parallelize the file-L3 `HiCacheFile.batch_get` + `batch_set` with a bounded ThreadPool**
(config `file_io_threads`, default serial=baseline). Lossless (disjoint files/host indices,
thread-safe evictor, unique tmp files). Screen with `mbench_file_io.py` on the node disk; if the
128-page batch speeds up >1.5x, full-eval. Rationale: disk backup+prefetch are on LooGLE's critical
path via a single serial thread; relieving it should speed host turnover + prefetch landing.
Next levers if v1 underwhelms: cache-aware queue ordering / anti-thrash admission; larger
chunked-prefill; write_back vs write_through.
