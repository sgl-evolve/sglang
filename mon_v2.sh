#!/usr/bin/env bash
# Capture v2 disk/GPU/metrics evidence with MINIMAL interference: detect serving by
# reading server.log on the shared FS (NO srun during init -- avoids disrupting the
# server's NCCL/c10d rendezvous), then do a few srun captures once serving.
set -uo pipefail
cd /home/junyanch_google_com/autoresearch/workspace/sgl/researchers/kv-lynx-4d2
LOG=eval-v2-scandirfix.log; SRV=runs/v2-scandirfix/server.log; OUT=disk_evidence_v2.txt
RT=/home/junyanch_google_com/autoresearch/programs/sgl/manager/.runtime
: > "$OUT"
# 1) wait for the node grab line
NODE=""
for i in $(seq 1 480); do
  NODE=$(grep -oE "node (slurm2-[a-z0-9-]+) OK" "$LOG" 2>/dev/null | tail -1 | awk '{print $2}')
  [ -n "$NODE" ] && break; sleep 15
done
[ -z "$NODE" ] && { echo "no node" >> "$OUT"; exit 1; }
JID=$(cat "$RT/held/$NODE" 2>/dev/null)
echo "v2 on node=$NODE jid=$JID at $(date '+%H:%M:%S')" >> "$OUT"
# 2) detect serving by reading server.log (no srun) -- scheduler batch activity
for i in $(seq 1 300); do
  if grep -qiE "Prefill batch|Decode batch|#running-req|gen throughput" "$SRV" 2>/dev/null; then
    echo "serving detected (server.log) at $(date '+%H:%M:%S')" >> "$OUT"; break
  fi
  sleep 15
done
sleep 150   # steady-state bursts
echo "=== iostat -x 5x8 (serving) $(date '+%H:%M:%S') ===" >> "$OUT"
timeout 60 srun --jobid=$JID --overlap -N1 -w $NODE bash -c \
  "iostat -x 5 8 2>/dev/null | grep -E 'Device|md127|nvme|dm-'" >> "$OUT" 2>&1
echo "=== GPU util (5 samples) ===" >> "$OUT"
timeout 25 srun --jobid=$JID --overlap -N1 -w $NODE bash -c \
  "for i in 1 2 3 4 5; do nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader|tr '\n' ' '; echo; sleep 2; done" >> "$OUT" 2>&1
echo "=== /metrics hicache snapshot ===" >> "$OUT"
timeout 20 srun --jobid=$JID --overlap -N1 -w $NODE bash -c \
  "curl -s http://localhost:30000/metrics 2>/dev/null | grep -iE 'prefetch|storage|hicache|queue|cache_hit|token_usage|num_running|num_waiting' | grep -v '^#' | head -70" >> "$OUT" 2>&1
echo "DONE $(date '+%H:%M:%S')" >> "$OUT"
