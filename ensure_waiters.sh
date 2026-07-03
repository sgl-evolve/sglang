#!/usr/bin/env bash
# ensure_waiters.sh — idempotent supervisor. Guarantees EXACTLY ONE node_waiter per usable held
# node. Safe to run every re-engagement: relaunches only MISSING waiters (never duplicates), and
# reports status (waiter wins / running evals / new completions / budget). One waiter per node.
set -uo pipefail
cd /home/junyanch_google_com/autoresearch/workspace/sgl/researchers/onyx-7q2
NODES="slurm2-a3nodeset-0 slurm2-a3nodeset1-2 slurm2-a3nodesetondem-3"
for nd in $NODES; do
  # count RUNNING waiters for this exact node (exclude this script + any grep)
  n=$(ps -eo args 2>/dev/null | grep -F "node_waiter.sh $nd" | grep -v grep | grep -cv ensure_waiters)
  if [ "$n" -eq 0 ]; then
    echo "[ensure] relaunch waiter $nd"
    nohup bash node_waiter.sh "$nd" > "waiter-$nd.out" 2>&1 &
  elif [ "$n" -gt 1 ]; then
    echo "[ensure] WARN $n waiters for $nd (dedup: keeping oldest)"
    # kill all but the oldest (smallest etime => oldest = last in default ps order; kill extras by pid)
    ps -eo pid,etimes,args | grep -F "node_waiter.sh $nd" | grep -v grep | grep -v ensure_waiters \
      | sort -k2 -n | head -n -1 | awk '{print $1}' | xargs -r kill 2>/dev/null
  else
    echo "[ensure] ok waiter $nd"
  fi
done
echo "--- wins/finishes ---"; grep -h "WON\|finished rc\|queue empty" waiter-*.out 2>/dev/null | tail -6 || true
echo "--- queue depth ---"; grep -cvE '^[[:space:]]*$' experiment_queue.txt 2>/dev/null
echo "--- logged versions (W&B run dirs) ---"; ls -d wandb/run-*onyx-7q2 2>/dev/null | wc -l
