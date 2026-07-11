# Mechanism design space — hybrid dual-cache reuse-frontier co-management

## Problem (confirmed)
Prefill-bound tail regime. Reuse depth = min(Full-KV frontier, Mamba frontier). Mamba state = 18 MB, scarce pool (1351 dev / 5360 host slots per rank); Full-KV host = 8.4 M tokens. The two pools evict **independently** → frontier misalignment → avoidable recompute → P99 TTFT tail (HoL blocking). Fixes must be "smarter engine, same memory" (resizing pools = banned budget/capacity).

## Key asymmetries
- **Stranding** (Mamba gone, Full present) is the harmful case: a turn-boundary checkpoint evicted under Mamba-pool pressure while its (large) Full KV survives in the roomier KV host pool → the Full KV becomes **un-continuable dead weight**, and the request recomputes its whole history.
- **Orphans** (Full gone, Mamba present) are rare: Full host eviction is leaf-atomic (`_evict_host_leaf` takes Mamba too).
- Binding-resource depends on workload: **many SHORT convs → Mamba-pool-bound** (5360 slots < #convs though Full fits); few LONG convs → Full-bound. Mooncake mix has many short ShareGPT convs ⇒ Mamba likely binding for that segment.

## Candidate mechanisms (pick by v1-instr data)
### M1 — Stranding-aware Full-KV reclamation (co-eviction) [novel primitive]
When a Mamba checkpoint at node N is evicted (host), the Full KV in N's subtree that is now un-continuable (below the next-shallower surviving Mamba checkpoint) is **dead weight**. Cascade-reclaim it → frees Full host space for *continuable* prefixes → higher effective hit, fewer Full evictions. Hooks: `MambaComponent.drive_host_eviction` / `evict_component(HOST)` → trigger Full reclamation of the stranded region. Lossless (reclaimed KV was un-reusable).

### M2 — Load-bearing Mamba protection (frontier-coupled retention)
In Mamba host eviction, deprioritize checkpoints whose Full KV is resident+hot; evict orphaned/cold first. Limited alone (few orphans) but composes with M1. Hook: `drive_host_eviction` victim selection.

### M3 — Value-density Mamba admission [if pool clutter exists]
Reserve scarce Mamba slots for checkpoints that unlock the most Full KV (deep prefixes / turn boundaries); skip caching low-value intermediate checkpoints. Hook: `commit_insert_component_data` / `should_skip_leaf_creation` for Mamba.

## Decision
- stranded≥5% & mamba_host_evict>0 → **M1 (primary) + M2**. This is the strong bet: co-eviction aligns the two frontiers, converting dead Full KV back into capacity.
- stranded≥5% & mamba_host_evict≈0 → sparse checkpointing; consider M3 / branch-point checkpointing.
- stranded<1% → pivot (recompute-tail scheduling / overlap).

## Eval plan (eval-frugal under contention)
v0-baseline (curve), v1-instr (stranding + replicate) → v2 mechanism (LAMPORT_INSTR on, show stranded→0) → 1 replicate + 1 ablation (M1 only vs M1+M2). Report full curve (p50/p90/p99 @ each λ) + recompute-token delta as the clean mechanism attribution.
