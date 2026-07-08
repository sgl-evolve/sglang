#!/usr/bin/env bash
set -uo pipefail
POOL=/home/junyanch_google_com/autoresearch/workspace/sgl/v0.25_ablations/_pool
AB=/home/junyanch_google_com/autoresearch/workspace/sgl/v0.25_ablations/base_free/researchers/base_free/ab_node.sh
export SGLANG_DG_CACHE_DIR=/mnt/localssd/base_free_dg
for att in $(seq 1 60); do
  for f in "$POOL"/held/*; do
    node=$(basename "$f"); jid=$(cat "$f" 2>/dev/null) || continue
    squeue -h -j "$jid" >/dev/null 2>&1 || continue
    exec 200>"$POOL/locks/$node.lock"
    if flock -n 200; then
      echo "[ab] attempt $att on $node"
      srun --jobid="$jid" --overlap -N1 -w "$node" --gres=gpu:8 bash "$AB" 3,4
      rc=$?; flock -u 200; exec 200>&-
      [ $rc -eq 0 ] && { echo "[ab] DONE"; exit 0; }
      echo "[ab] rc=$rc retry in 40s"; sleep 40; break
    fi
    exec 200>&-
  done
  sleep 25
done
echo "[ab] gave up"
