# Related-work / prior-art notes (from repo docs + code)

## HiCache write-back policies (docs/advanced_features/hicache_design.md)
- write_through (eval baseline): every access immediately written back. "When bandwidth is
  sufficient, this strategy provides the strongest caching benefit."
- write_through_selective: written back only after access FREQUENCY exceeds a threshold
  ("backs up only hot data, reducing I/O overhead"). == my `flat` reuse-gate (threshold=2). CONFIG.
- write_back: written back only on eviction from upper tier ("suitable when storage capacity is
  limited but memory utilization must be maximized").
=> Reuse-gated admission is already a stock config (write_through_selective). Any admission-only
   contribution is config-equivalent. The designers' own note ("write_through best when bandwidth
   sufficient") predicts selective HELPS ONLY when backup bandwidth is scarce — which it ISN'T here
   (~1 GB/s vs 64+ GB/s hw). So flat/selective likely <= stock on this workload. TEST confirms.

## Prefetch (L3 only; N/A here — 2-tier, no L3)
best_effort / wait_complete / timeout strategies gate L2<-L3 prefetch. We have no L3.

## Positioning for a novel contribution
- Eviction: LRU~Belady (dead end, confirmed by siblings + my oracle analysis).
- Admission: config-reachable (write_through_selective); bandwidth-free => likely no win.
- Real binding constraint (from v1_stock server.log): at λ=3 concurrency is already MAXED (256),
  device KV 84-100% full of running-request state. Prefix cache is L2-bound; L2 hit 0.62 vs 0.809
  oracle. Headroom is capacity-bound (frozen budget) + info-constrained (can't predict one-shot at
  turn 0). => a big lossless *caching* win is unlikely; novelty must target a different axis
  (giant turn-0 prefill TTFT tail; transfer/compute under device pressure; a new primitive).
