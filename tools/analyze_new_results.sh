#!/usr/bin/env bash
# Quick analysis for newly completed evals.
# Usage: bash tools/analyze_new_results.sh <version_name>
set -euo pipefail
WS="$(cd "$(dirname "$0")/.." && pwd)"
cd "$WS"

VER="${1:?usage: analyze_new_results.sh <version_name>}"
SUMMARY="runs/$VER/summary.json"
RESOLVED="runs/$VER/resolved_args.json"

if [ ! -f "$SUMMARY" ]; then echo "ERROR: $SUMMARY not found"; exit 1; fi

echo "=== $VER ==="
# Key metrics
.venv/bin/python -c "
import json
s = json.load(open('$SUMMARY'))
m = s.get('mix', {})
print(f\"hit_rate:       {m.get('hit_rate', '?')}\")
print(f\"p50_ms:         {m.get('ttft_median_ms', m.get('ttft_p50_ms', '?'))}\")
print(f\"p99_ms:         {m.get('ttft_p99_ms', '?')}\")
print(f\"mean_ms:        {m.get('ttft_mean_ms', '?')}\")
print(f\"req_per_s:      {m.get('req_throughput', m.get('req_per_s', '?'))}\")
print(f\"host_util:      {m.get('hicache_host_util', '?')}\")
print(f\"evict_mean_ms:  {m.get('hicache_eviction_mean_ms', '?')}\")
print(f\"load_back_ms:   {m.get('hicache_load_back_mean_ms', '?')}\")
"

# Check resolved args for correctness
if [ -f "$RESOLVED" ]; then
  echo "--- resolved_args ---"
  .venv/bin/python -c "
import json
d = json.load(open('$RESOLVED'))
print(f\"write_policy:   {d.get('hicache_write_policy', '?')}\")
print(f\"eviction:       {d.get('radix_eviction_policy', 'default(lru)')}\")
print(f\"schedule:       {d.get('schedule_policy', 'default(fcfs)')}\")
"
fi

# z-test vs LRU reference
echo "--- z-test vs LRU (n=6, μ=0.7520, σ=0.00033) ---"
.venv/bin/python -c "
import json
s = json.load(open('$SUMMARY'))
h = s.get('mix', {}).get('hit_rate')
if h is None: print('NO HIT RATE'); exit()
mu, sigma = 0.7520, 0.00033
z = (h - mu) / sigma if sigma > 0 else 0
sig = 'NS' if abs(z) < 2 else ('WORSE' if z < 0 else 'BETTER')
print(f'hit={h:.4f}  z={z:.2f}  {sig}  Δ={(h-mu)*100:+.2f}pp')
"
