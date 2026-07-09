#!/usr/bin/env bash
POOL="/home/junyanch_google_com/autoresearch/workspace/sgl/v0.25_ablations/_pool"
while true; do
  echo "[$(date -u '+%H:%M:%SZ')]"
  for f in "$POOL/held"/*; do
    [ -f "$f" ] || continue
    node=$(basename "$f"); jid=$(cat "$f")
    squeue -h -j "$jid" >/dev/null 2>&1 || { echo "  $node: EXPIRED"; continue; }
    lockfile="$POOL/locks/$node.lock"
    exec 200>"$lockfile"
    if flock -n 200; then
      echo "  $node: FREE (no flock!)"
      flock -u 200
    else
      reqs=$(timeout 5 srun --jobid="$jid" --overlap -N1 -w "$node" bash -c 'curl -s http://localhost:30000/metrics 2>/dev/null | grep "sglang:num_requests_total{" | head -1 | awk "{print \$NF}"' 2>&1 || echo "?")
      echo "  $node: BUSY reqs=$reqs/7037"
    fi
    exec 200>&-
  done
  # Check if any of my evals completed
  new=$(find /home/junyanch_google_com/autoresearch/workspace/sgl/v0.25_ablations/sgl_free/researchers/sgl_free/runs -name "summary.json" -newer /tmp/.sgl_free_last_check 2>/dev/null | wc -l)
  touch /tmp/.sgl_free_last_check
  [ "$new" -gt 0 ] && echo "  *** $new NEW completions! ***"
  sleep 180
done
