#!/usr/bin/env bash
# Log all new eval versions that have summary.json but haven't been logged yet.
# Usage: bash log_all_new.sh
set -uo pipefail
ROOT="${AUTORESEARCH_ROOT:-/home/junyanch_google_com/autoresearch}"
[ -f "$ROOT/.env" ] && { set -a; . "$ROOT/.env"; set +a; }
export HF_TOKEN="$HF_API_KEY"

SGL_HOME="/home/junyanch_google_com/autoresearch/programs/sgl/v0.25_ablations/sgl_mech"
LOGGER="$SGL_HOME/researcher/.claude/skills/report-sop/scripts/log_wandb.py"
WS="/home/junyanch_google_com/autoresearch/workspace/sgl/v0.25_ablations/sgl_mech/researchers/sgl_mech"
NAME="sgl_mech"

source "$WS/.venv/bin/activate"
export PYTHONPATH="$WS/python"

# List of new versions from the ablation batch
NEW_VERSIONS=(
    v9-gdsf v10-lfu v11-slru v12-costfreq v13-contcost
    v14-writeadmit v15-sjf v16-warmfirst v17-freqdecay v18-sizelru
    v19-loadback v0-ctl3
    v21-valuegate v22-wt2 v23-wt3 v24-lru-valuegate v25-fullstack v26-lru-wt2
    v27-backupcost v28-freqcost v29-contcost-a50 v30-contcost-a200
    v31-reuse1 v32-t4096 v33-backupcost-wt2 v34-3tier v35-freqcost-w20 v0-ctl4
    v36-splittier-lru v37-splittier-gdsf v38-lru-splittier-cost v0-ctl5 v39-t2048-ctl
)

COMMIT=$(cd "$WS" && git rev-parse --short HEAD 2>/dev/null || echo "?")

for ver in "${NEW_VERSIONS[@]}"; do
    summary="$WS/runs/$ver/summary.json"
    if [ -f "$summary" ]; then
        echo ">>> Logging $ver (commit $COMMIT) as mechanism..."
        python3 "$LOGGER" "$NAME" "$summary" "$ver" "$COMMIT" mechanism
        echo "    logged $ver"
    else
        echo "--- Skipping $ver (no summary.json yet)"
    fi
done

# Log v0-ctl3 as baseline (config) if present
ctl3="$WS/runs/v0-ctl3/summary.json"
if [ -f "$ctl3" ]; then
    echo ">>> Re-logging v0-ctl3 as baseline mechanism..."
    # Already logged above as mechanism, which is fine — it's a control run with the mechanism code
fi

echo "=== Done ==="
