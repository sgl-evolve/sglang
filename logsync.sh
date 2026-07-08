#!/usr/bin/env bash
# Log a version to W&B via OFFLINE mode + sync (online wandb.init times out from the
# login node; `wandb sync` of an offline run works). Usage:
#   bash logsync.sh <version> <commit> <config|mechanism>   # uses runs/<version>/summary.json
set -uo pipefail
cd /home/junyanch_google_com/autoresearch/workspace/sgl/v0.25_ablations/base_free/researchers/base_free
_SGL_HOME="$SGL_HOME"; _SGL_WS="$SGL_WORKSPACE"
set -a; source /home/junyanch_google_com/autoresearch/.env; set +a
export SGL_HOME="$_SGL_HOME"; export SGL_WORKSPACE="$_SGL_WS"
export WANDB_MODE=offline
source .venv/bin/activate
VER="${1:?version}"; COMMIT="${2:?commit}"; TAG="${3:?config|mechanism}"
SUM="runs/$VER/summary.json"
[ -f "$SUM" ] || { echo "no $SUM"; exit 1; }
python "$SGL_HOME/researcher/.claude/skills/report-sop/scripts/log_wandb.py" base_free "$SUM" "$VER" "$COMMIT" "$TAG" 2>&1 | tail -4
# sync the newest offline run dir
d=$(ls -dt wandb/offline-run-*base_free 2>/dev/null | head -1)
[ -n "$d" ] && { echo "syncing $d"; timeout 180 wandb sync "$d" 2>&1 | tail -3; }
