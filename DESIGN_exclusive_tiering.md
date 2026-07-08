# Mechanism design: EXCLUSIVE L1↔L2 KV-cache tiering

## Problem (evidence-backed)
- Both tiers full; hit 0.62; working set 19M >> cache. System on the STEEP part of the hit-vs-capacity
  curve (sim: C 7.5→10.2M ⇒ hit 0.45→0.73). So EFFECTIVE capacity is the dominant lever.
- Under stock `write_through` (default), HiCache is INCLUSIVE: `_inc_hit_count` (urc.py:1819) eagerly
  calls `write_backup` on first reuse and KEEPS the device copy ⇒ every hot device entry is duplicated on
  host ⇒ distinct capacity ≈ HOST alone (7.81M). Real baseline hit 0.62 ≈ sim at C≈8M ⇒ confirms device
  (2.35M) is largely redundant.
- `v0_lpm` proved ordering can't recover hit (0.62→0.62, p99 21s). Charter: eviction-order is a dead end
  (LRU≈Belady). So the recoverable headroom is EFFECTIVE CAPACITY via L1↔L2 placement (charter-listed).

## The two exclusivity halves
1. **Write-side (stock `write_back` already does this):** `_inc_hit_count` RETURNS immediately for
   write_back (urc.py:1815-1817) ⇒ NO eager backup. Entry stays device-only until device eviction, then
   `_evict_device_leaf` write_back path (urc.py:1497-1503) writes it to host and frees device ⇒ host-only.
   ⇒ device XOR host on the write path. **Tested by the `s_writeback` screen (running).**
2. **Promotion-side (the NOVEL engine bit):** on a host hit, `load_back` (urc.py:1660) copies H→D but
   `loading_check` (urc.py:2382) only drops lock-refs — it does NOT free the host copy ⇒ promoted entries
   become INCLUSIVE again (device+host). **Fix:** after the load completes (finish_event synchronized in
   loading_check), FREE the node's host_value ⇒ fully device-exclusive. Reclaims host capacity for other
   distinct entries.

## Mechanism v1 = write_back + free-host-on-promotion (env `SGLANG_HICACHE_EXCLUSIVE`, default off)
- Requires write_back semantics (or fold the write-side in). The added code: in `loading_check`, for each
  completed load-back node, free its host component values (KV + mamba host), update host-LRU, set
  backuped→False. Lossless: device copy present before host freed; later device eviction re-backs-up to
  host (write_backup handles parent-first invariant). Match still works: node matches via device value
  (validators allow value-or-host_value), mixed device/host paths already supported.

## Correctness hazards to verify before eval
- Free host only AFTER finish_event.synchronize() (device copy committed) — else corruption.
- Don't free host_value of a node still needed by an ongoing write-back/other op (check ongoing_* dicts).
- Host radix invariant: freeing a node's host copy while it's device-resident is OK (match via device);
  but on later re-eviction write_backup must re-establish parent host backup (it recurses — OK).
- Mamba component: free MAMBA host state consistently with FULL (per-component free).

## Expected result (sim-predicted, to confirm)
- Baseline (inclusive, C≈host 7.81M): hit ~0.62.
- write_back (write-side exclusive): hit toward ~0.68-0.72 (partial — promotion re-duplicates).
- write_back + free-host-on-promotion (full exclusive, C→10.16M): hit ~0.72+ ⇒ lower fresh prefill ⇒
  lower p99 TTFT tail ⇒ goodput curve shifts up. Verify with λ sweep (3/4/5).
- Attribution: v0_official (inclusive) vs write_back (write-excl) vs full-excl isolates each half.

## Decision gate
- If `s_writeback` hit_rate > baseline ⇒ exclusivity confirmed ⇒ build+eval full mechanism (this doc).
- If `s_writeback` ≈ baseline ⇒ premise wrong (device not actually redundant, or promotion dominates) ⇒
  reconsider (maybe promotion-side is the whole story ⇒ still build free-host-on-promotion; or pivot).
