#!/usr/bin/env bash
# Wait until the v1-timeout server is serving, then capture disk iostat + a /metrics
# snapshot from the eval node (job 18104, node slurm2-a3nodeset1-2). Non-perturbing
# (iostat reads /proc; one curl). Writes to disk_evidence.txt.
set -uo pipefail
cd /home/junyanch_google_com/autoresearch/workspace/sgl/researchers/kv-lynx-4d2
JID=18104; NODE=slurm2-a3nodeset1-2; OUT=disk_evidence.txt
: > "$OUT"
# 1) wait for server ready (/health 200) up to ~30 min
for i in $(seq 1 180); do
  ok=$(timeout 15 srun --jobid=$JID --overlap -N1 -w $NODE bash -c \
        "curl -s -o /dev/null -w '%{http_code}' http://localhost:30000/health 2>/dev/null" 2>/dev/null)
  if [ "$ok" = "200" ]; then echo "server ready at $(date '+%H:%M:%S')" >> "$OUT"; break; fi
  sleep 10
done
# 2) let the bench ramp for 90s so we catch steady-state bursts, then sample iostat
sleep 90
echo "=== iostat -x 5 x8 (during serving) $(date '+%H:%M:%S') ===" >> "$OUT"
timeout 60 srun --jobid=$JID --overlap -N1 -w $NODE bash -c \
  "iostat -x 5 8 md127 2>/dev/null || iostat -x 5 8 2>/dev/null | grep -E 'Device|md127|nvme'" >> "$OUT" 2>&1
echo "=== /metrics hicache snapshot $(date '+%H:%M:%S') ===" >> "$OUT"
timeout 20 srun --jobid=$JID --overlap -N1 -w $NODE bash -c \
  "curl -s http://localhost:30000/metrics 2>/dev/null | grep -iE 'prefetch|storage|hicache|queue|cache_hit|token_usage' | grep -v '^#' | head -60" >> "$OUT" 2>&1
echo "DONE $(date '+%H:%M:%S')" >> "$OUT"
