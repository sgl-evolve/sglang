#!/usr/bin/env bash
# monitor_batch.sh — monitor batch_eval.sh progress and auto-log completed results.
# Run this in a separate terminal alongside batch_eval.sh.
set -uo pipefail
cd /home/junyanch_google_com/autoresearch/workspace/sgl/v0.25_ablations/base_free/researchers/base_free

_SGL_HOME="$SGL_HOME"; _SGL_WS="$SGL_WORKSPACE"
set -a; source /home/junyanch_google_com/autoresearch/.env; set +a
export SGL_HOME="$_SGL_HOME"; export SGL_WORKSPACE="$_SGL_WS"
export WANDB_API_KEY
export HF_TOKEN="$HF_API_KEY"

LOGGER="$SGL_HOME/researcher/.claude/skills/report-sop/scripts/log_wandb.py"
source .venv/bin/activate

LOGGED_FILE="/tmp/base_free_logged_versions.txt"
touch "$LOGGED_FILE"

classify() {
  local ver="$1"
  case "$ver" in
    v*-cfg-*) echo "config" ;;
    v*-baseline*|v26-*|v27-*|diag-*) echo "config" ;;
    *) echo "mechanism" ;;
  esac
}

echo "=== monitor_batch.sh started $(date) ==="
while true; do
  total=0; done=0; pending=0
  for dir in runs/v*/; do
    ver=$(basename "$dir")
    total=$((total+1))
    if [[ -f "$dir/summary.json" ]]; then
      done=$((done+1))
      # Auto-log to W&B if not already logged
      if ! grep -qF "$ver" "$LOGGED_FILE" 2>/dev/null; then
        commit=$(python3 -c "import json; print(json.load(open('$dir/summary.json')).get('commit','?'))" 2>/dev/null || echo "?")
        kind=$(classify "$ver")
        echo "[$(date +%H:%M)] logging $ver (commit=$commit, kind=$kind)"
        python3 "$LOGGER" base_free "$dir/summary.json" "$ver" "$commit" "$kind" 2>&1 | tail -3
        echo "$ver" >> "$LOGGED_FILE"
      fi
    else
      pending=$((pending+1))
    fi
  done
  echo "[$(date +%H:%M)] progress: $done/$total done, $pending pending"

  # Also count non-v (diag, cfg) versions
  for dir in runs/diag-*/ runs/cfg-*/; do
    [[ -d "$dir" ]] || continue
    ver=$(basename "$dir")
    if [[ -f "$dir/summary.json" ]] && ! grep -qF "$ver" "$LOGGED_FILE" 2>/dev/null; then
      commit=$(python3 -c "import json; print(json.load(open('$dir/summary.json')).get('commit','?'))" 2>/dev/null || echo "?")
      kind=$(classify "$ver")
      echo "[$(date +%H:%M)] logging $ver (commit=$commit, kind=$kind)"
      python3 "$LOGGER" base_free "$dir/summary.json" "$ver" "$commit" "$kind" 2>&1 | tail -3
      echo "$ver" >> "$LOGGED_FILE"
    fi
  done

  # Sync W&B
  bash logsync.sh 2>&1 | tail -2

  sleep 120
done
