#!/usr/bin/env bash
# Wait for v2 to grab a node + start serving, then capture disk iostat + GPU util +
# /metrics from that node. Discovers node & hold-jobid from the run_eval log.
set -uo pipefail
cd /home/junyanch_google_com/autoresearch/workspace/sgl/researchers/kv-lynx-4d2
LOG=eval-v2-scandirfix.log; OUT=disk_evidence_v2.txt; RT=/home/junyanch_google_com/autoresearch/programs/sgl/manager/.runtime
: > "$OUT"
# 1) wait until run_eval prints the chosen node
NODE=""
for i in $(seq 1 360); do
  NODE=$(grep -oE "node (slurm2-[a-z0-9-]+) OK" "$LOG" 2>/dev/null | tail -1 | awk '{print $2}')
  [ -n "$NODE" ] && break; sleep 15
done
[ -z "$NODE" ] && { echo "no node found" >> "$OUT"; exit 1; }
JID=$(cat "$RT/held/$NODE" 2>/dev/null)
echo "v2 on node=$NODE jid=$JID at $(date '+%H:%M:%S')" >> "$OUT"
# 2) wait for /health 200
for i in $(seq 1 240); do
  ok=$(timeout 15 srun --jobid=$JID --overlap -N1 -w $NODE bash -c \
        "curl -s -o /dev/null -w '%{http_code}' http://localhost:30000/health 2>/dev/null" 2>/dev/null)
  [ "$ok" = "200" ] && { echo "serving at $(date '+%H:%M:%S')" >> "$OUT"; break; }
  sleep 10
done
sleep 120   # let the bench reach steady-state bursts
echo "=== iostat -x 5x8 md127 (serving) $(date '+%H:%M:%S') ===" >> "$OUT"
timeout 60 srun --jobid=$JID --overlap -N1 -w $NODE bash -c \
  "iostat -x 5 8 2>/dev/null | grep -E 'Device|md127|nvme|dm-' " >> "$OUT" 2>&1
echo "=== GPU util samples ===" >> "$OUT"
timeout 30 srun --jobid=$JID --overlap -N1 -w $NODE bash -c \
  "for i in 1 2 3 4 5; do nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader | tr '\n' ' '; echo; sleep 2; done" >> "$OUT" 2>&1
echo "=== /metrics hicache snapshot ===" >> "$OUT"
timeout 20 srun --jobid=$JID --overlap -N1 -w $NODE bash -c \
  "curl -s http://localhost:30000/metrics 2>/dev/null | grep -iE 'prefetch|storage|hicache|queue|cache_hit|token_usage|num_running|num_waiting' | grep -v '^#' | head -70" >> "$OUT" 2>&1
echo "DONE $(date '+%H:%M:%S')" >> "$OUT"
