#!/usr/bin/env bash
# log_new_versions.sh — log all completed runs to W&B that haven't been logged yet.
set -uo pipefail
SGL_HOME=/home/junyanch_google_com/autoresearch/programs/sgl/v0.25_ablations/base_mech
WS=/home/junyanch_google_com/autoresearch/workspace/sgl/v0.25_ablations/base_mech/researchers/base_mech
LOGGER="$SGL_HOME/researcher/.claude/skills/report-sop/scripts/log_wandb.py"
NAME=base_mech

set -a; source /home/junyanch_google_com/autoresearch/.env; set +a
export WANDB_API_KEY HF_TOKEN="$HF_API_KEY"
source "$WS/.venv/bin/activate"
cd "$WS"

# Track logged versions
LOGGED_FILE="$WS/.logged_versions"
touch "$LOGGED_FILE"

for d in "$WS/runs/"*/; do
  ver=$(basename "$d")
  [ -f "$d/summary.json" ] || continue
  grep -qx "$ver" "$LOGGED_FILE" 2>/dev/null && continue
  commit=$(python3 -c "import json; d=json.load(open('$d/summary.json')); print(d.get('commit','?'))" 2>/dev/null)
  echo "Logging $ver (commit=$commit)..."
  python3 "$LOGGER" "$NAME" "$d/summary.json" "$ver" "${commit:-?}" mechanism
  rc=$?
  if [ $rc -eq 0 ]; then
    echo "$ver" >> "$LOGGED_FILE"
    echo "  -> logged OK"
  else
    echo "  -> FAILED (rc=$rc)"
  fi
done
echo "=== sync offline runs ==="
for d in wandb/offline-run-*; do
  [ -d "$d" ] || continue
  wandb sync "$d" 2>&1 | tail -1
done
