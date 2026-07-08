#!/usr/bin/env bash
# sweep_retry.sh <label> [extra server args...]
# Grab a DRAM-healthy held pool node (flock, retry past base_free's busy/low-DRAM nodes) and srun the
# goodput-curve sweep (sweep_onnode.sh) onto it. SGLANG_* env is inherited (--export=ALL). Blocks until a
# curve.csv is produced. Run in background.
set -uo pipefail
ROOT=/home/junyanch_google_com/autoresearch
POOL="$ROOT/workspace/sgl/v0.25_ablations/_pool"
WS="$ROOT/workspace/sgl/v0.25_ablations/sgl_free/researchers/sgl_free"
LABEL="${1:?usage: sweep_retry.sh <label> [args]}"; shift || true
OUT="$WS/runs/sweep_$LABEL"; CURVE="$OUT/curve.csv"
held_nodes(){ for f in "$POOL"/held/*; do [ -e "$f" ] && basename "$f"; done; }
echo "[sweep_retry] label=$LABEL args=[$*] exclusive=[${SGLANG_HICACHE_EXCLUSIVE:-unset}]"
for try in $(seq 1 200); do
  [ -f "$CURVE" ] && { echo "[sweep_retry] curve exists -> done"; break; }
  for node in $(held_nodes); do
    jid=$(cat "$POOL/held/$node" 2>/dev/null) || continue
    squeue -h -j "$jid" >/dev/null 2>&1 || continue
    exec 200>"$POOL/locks/$node.lock"
    if flock -n 200; then
      echo "[sweep_retry] try $try got $node (job $jid) $(date -u +%H:%M:%S)"
      srun --jobid="$jid" --overlap -N1 -w "$node" --gres=gpu:8 \
        bash "$WS/tools/sweep_onnode.sh" "$LABEL" "$@"
      rc=$?
      flock -u 200; exec 200>&-
      [ -f "$CURVE" ] && { echo "[sweep_retry] SUCCESS (rc=$rc)"; break 2; }
      echo "[sweep_retry] try $try ended rc=$rc no curve; retry"
      break
    fi
    exec 200>&-
  done
  sleep 45
done
[ -f "$CURVE" ] && { echo "[sweep_retry] DONE:"; cat "$CURVE"; } || echo "[sweep_retry] gave up"
