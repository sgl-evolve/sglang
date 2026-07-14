#!/usr/bin/env bash
# When v5_size finishes: log to W&B, release the held node.
set -uo pipefail
cd /home/junyanch_google_com/autoresearch/workspace/sgl/v0.31/research/researchers/turing
L=/tmp/turing_fin_size.log
echo "[fin] wait v5_size summary $(date -u +%H:%M:%S)" > "$L"
for i in $(seq 1 300); do [ -f runs/v5_size/summary.json ] && break; sleep 60; done
echo "[fin] v5_size done; logging $(date -u +%H:%M:%S)" >> "$L"
source .venv/bin/activate 2>/dev/null; set -a; . /home/junyanch_google_com/autoresearch/.env 2>/dev/null; set +a
python3 /home/junyanch_google_com/autoresearch/programs/sgl/v0.31/research/researcher/.claude/skills/report-sop/scripts/log_wandb.py \
  turing runs/v5_size/summary.json v5_size ce01c1c79 mechanism >> "$L" 2>&1
scancel "$(cat /tmp/turing_hold_jid.txt)" >> "$L" 2>&1
echo "[fin] logged + released $(date -u +%H:%M:%S)" >> "$L"
