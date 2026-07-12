#!/usr/bin/env bash
# DECISIVE scheduler-cache COUPLING experiment (§2.4 confirmation), same-node on idle held ondem-2.
# Order = most-decisive first, so partial completion still yields the key comparison:
#   1. stock_fcfs : PIN off, default fcfs            (anchor, same-node)
#   2. stock_lpm  : PIN off, --schedule-policy lpm   (COUPLING: does LPM raise hit + change p99 tail?)
#   3. v1_lpm     : PIN on,  --schedule-policy lpm   (does pin still help under LPM? sim predicts ~0)
#   4. v1_fcfs    : PIN on,  default fcfs            (same-node re-anchor of the +2.5pp win)
# nohup'd (survives session boundaries); flock ondem-2 so no collision; serial (avoids flashinfer JIT race).
set -uo pipefail
ROOT=/home/junyanch_google_com/autoresearch
EVAL=$ROOT/programs/sgl/v0.31/research/researcher/.claude/skills/evaluation-sop/scripts/eval.sh
POOL=$ROOT/workspace/sgl/v0.31/_pool
NODE=slurm2-a3nodesetondem-2
JID=$(cat "$POOL/held/$NODE" 2>/dev/null)
LOG=$ROOT/workspace/sgl/v0.31/research/researchers/valiant
run(){ # <version> <pin> <extra...>
  local ver="$1" pin="$2"; shift 2
  # skip if a completed curve already exists (robust to restart)
  if [ -s "$LOG/runs/$ver/curve.csv" ] && [ "$(wc -l < "$LOG/runs/$ver/curve.csv")" -ge 5 ]; then
    echo "[coup] $(date -u +%H:%M:%S) SKIP $ver (curve.csv already complete)"; return 0; fi
  echo "[coup] $(date -u +%H:%M:%S) START $ver (pin=$pin extra=$*)"
  env VALIANT_PIN_ENABLE=$pin srun --jobid=$JID --overlap -N1 -w $NODE --gres=gpu:8 \
      bash "$EVAL" valiant "$ver" "$@" > "$LOG/eval-$ver.log" 2>&1
  echo "[coup] $(date -u +%H:%M:%S) END $ver rc=$?"
}
exec 200>"$POOL/locks/$NODE.lock"
flock -n 200 || { echo "[coup] $NODE busy (locked by sibling); abort"; exit 1; }
echo "[coup] locked $NODE jid=$JID, pipeline start $(date -u +%H:%M:%S)"
run stock_fcfs 0
run stock_lpm  0 --schedule-policy lpm
run v1_lpm     1 --schedule-policy lpm
run v1_fcfs    1
flock -u 200
echo "[coup] pipeline DONE $(date -u +%H:%M:%S)"
