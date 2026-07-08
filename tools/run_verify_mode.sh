#!/usr/bin/env bash
# flock a pool node (optionally excluding one), srun the lossless verify in a given mode. retry.
set -uo pipefail
MODE="${1:?mode}"; EXCLUDE="${2:-}"
ROOT=/home/junyanch_google_com/autoresearch
POOL="$ROOT/workspace/sgl/v0.25_ablations/_pool"
WS="$ROOT/workspace/sgl/v0.25_ablations/sgl_free/researchers/sgl_free"
ON="$WS/tools/lossless_verify_onnode.sh"
held_nodes(){ for f in "$POOL"/held/*; do [ -e "$f" ] && basename "$f"; done; }
for try in $(seq 1 200); do
  [ -f "$WS/runs/lossless_verify/outputs_$MODE.json" ] && { echo "[v] $MODE done"; break; }
  for node in $(held_nodes); do
    [ "$node" = "$EXCLUDE" ] && continue
    jid=$(cat "$POOL/held/$node" 2>/dev/null) || continue
    squeue -h -j "$jid" >/dev/null 2>&1 || continue
    exec 200>"$POOL/locks/$node.lock"
    if flock -n 200; then
      echo "[v] try $try got $node $(date -u +%H:%M:%S) mode=$MODE"
      srun --jobid="$jid" --overlap -N1 -w "$node" --gres=gpu:8 bash "$ON" "$MODE"
      rc=$?; flock -u 200; exec 200>&-
      [ -f "$WS/runs/lossless_verify/outputs_$MODE.json" ] && { echo "[v] SUCCESS"; break 2; }
      echo "[v] try $try no output (rc=$rc); retry"; break
    fi
    exec 200>&-
  done
  sleep 45
done
