#!/usr/bin/env bash
# Detached, session-independent good-neighbor waiter: launch lpm@1553 on first genuinely-idle
# held node (>=1.3TB, flock-free), write result. One probe then exit. Runs up to ~12h.
POOL=/home/junyanch_google_com/autoresearch/workspace/sgl/v0.3/_pool
W=/home/junyanch_google_com/autoresearch/workspace/sgl/v0.3/research/researchers/lamport
DIAG="$W/tools/diag_pressure.sh"
LOG="$W/probe-lpm.log"
echo "[persist-lpm] start $(date -u +%H:%M:%S)" >> "$LOG"
for i in $(seq 1 700); do
  [ -f "$W/runs/diag1553lpm/curve.csv" ] && grep -q sweep "$W/runs/diag1553lpm/curve.csv" 2>/dev/null && { echo "[persist-lpm] already done" >> "$LOG"; exit 0; }
  for node in slurm2-a3nodesetondem-2 slurm2-a3nodeset0-3 slurm2-a3nodeset1-2; do
    jid=$(squeue -h -w "$node" -o "%i" 2>/dev/null | grep -E "^[0-9]+$" | head -1)
    [ -z "$jid" ] && continue
    mem=$(timeout 20 srun --jobid="$jid" --overlap -N1 -w "$node" awk '/MemAvailable/{printf "%d",$2/1048576}' /proc/meminfo 2>/dev/null)
    if [ "${mem:-0}" -ge 1300 ]; then
      exec 200>"$POOL/locks/$node.lock"
      if flock -n 200; then
        echo "[persist-lpm] launching lpm@1553 on $node (${mem}G) $(date -u +%H:%M:%S)" >> "$LOG"
        srun --jobid="$jid" --overlap -N1 -w "$node" --gres=gpu:8 bash "$DIAG" diag1553lpm 1553 "5" --schedule-policy lpm >> "$LOG" 2>&1
        echo "[persist-lpm] rc=$? done $(date -u +%H:%M:%S)" >> "$LOG"
        flock -u 200; exit 0
      fi
      exec 200>&-
    fi
  done
  sleep 45
done
echo "[persist-lpm] timeout (~12h), pool stayed saturated $(date -u +%H:%M:%S)" >> "$LOG"
