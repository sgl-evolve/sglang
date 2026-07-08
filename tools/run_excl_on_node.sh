#!/usr/bin/env bash
# Run ONE exclusive eval on a SPECIFIC node (to pair with an existing same-node baseline). retry on crash.
set -uo pipefail
NODE="${1:?node}"; VER="${2:?ver}"
ROOT=/home/junyanch_google_com/autoresearch
POOL="$ROOT/workspace/sgl/v0.25_ablations/_pool"
WS="$ROOT/workspace/sgl/v0.25_ablations/sgl_free/researchers/sgl_free"
EVAL="$ROOT/programs/sgl/v0.25_ablations/sgl_free/researcher/.claude/skills/evaluation-sop/scripts/eval.sh"
export SGLANG_HICACHE_EXCLUSIVE=1
for try in $(seq 1 4); do
  [ -f "$WS/runs/$VER/summary.json" ] && { echo "[x] done"; break; }
  jid=$(cat "$POOL/held/$NODE" 2>/dev/null) || { echo "[x] $NODE not held"; sleep 45; continue; }
  squeue -h -j "$jid" >/dev/null 2>&1 || { echo "[x] hold $jid dead"; sleep 60; continue; }
  exec 200>"$POOL/locks/$NODE.lock"
  if flock -n 200; then
    echo "[x] try $try got $NODE (job $jid) $(date -u +%H:%M:%S)"
    srun --jobid="$jid" --overlap -N1 -w "$NODE" --gres=gpu:8 bash "$EVAL" sgl_free "$VER"
    rc=$?; flock -u 200; exec 200>&-
    [ -f "$WS/runs/$VER/summary.json" ] && { echo "[x] SUCCESS"; break; }
    echo "[x] try $try no summary (rc=$rc); retry after 60s"; sleep 60
  else
    exec 200>&-; echo "[x] $NODE flock busy; wait"; sleep 45
  fi
done
python3 -c "import json;p=json.load(open('$WS/runs/$VER/summary.json'))['panel'];print('$VER: hit',round(p['overall/hit_rate'],4),'p99',p['overall/ttft_p99_ms'],'mean',p['overall/ttft_mean_ms'])" 2>/dev/null || echo "[x] no result"
