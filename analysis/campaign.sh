#!/usr/bin/env bash
# floyd serial same-node A/B campaign driver.
# Runs a list of evals SEQUENTIALLY (never 2 floyd evals at once -> avoids the
# shared-NFS-cache wedge), pinned to ONE certified node (-w) --exclusive so the
# stock-vs-CCA comparison is same-node. Each eval is sbatch --wait (blocks).
# Resumable: skips a version whose runs/<ver>/summary.json already exists.
#
# Usage: nohup bash analysis/campaign.sh <node> > runs/campaign.log 2>&1 &
set -uo pipefail
ROOT=/home/junyanch_google_com/autoresearch
SGL=$ROOT/programs/sgl/v0.31/research
EVAL=$SGL/researcher/.claude/skills/evaluation-sop/scripts/eval.sh
WORK=$ROOT/workspace/sgl/v0.31/research/researchers/floyd
LOGD=$SGL/manager/.runtime/logs
NODE="${1:-slurm2-a3nodesetondem-3}"

# version : extra eval flags  (stock has no flags). Same-node A/B, interleaved.
JOBS=(
  "v1-cca:--enable-cca-prefill --cca-watermark 0.85 --cca-threshold 4096 --cca-max-defer-ms 4000"
  "v0-stock-r2:"
  "v1-cca-r2:--enable-cca-prefill --cca-watermark 0.85 --cca-threshold 4096 --cca-max-defer-ms 4000"
  "v0-stock-r3:"
  "v1-cca-r3:--enable-cca-prefill --cca-watermark 0.85 --cca-threshold 4096 --cca-max-defer-ms 4000"
)

echo "[campaign] node=$NODE  $(date -u)"
for entry in "${JOBS[@]}"; do
  ver="${entry%%:*}"; flags="${entry#*:}"
  if [ -f "$WORK/runs/$ver/summary.json" ]; then
    echo "[campaign] SKIP $ver (summary.json exists)"; continue
  fi
  log="$LOGD/eval-floyd-$ver-%j.out"; mkdir -p "$LOGD" 2>/dev/null || log="/tmp/eval-floyd-$ver-%j.out"
  echo "[campaign] >>> $ver  flags='$flags'  $(date -u)"
  # shellcheck disable=SC2086
  sbatch --wait -p a3 -N1 --exclusive --gres=gpu:8 -w "$NODE" -t 6:00:00 \
    -J "eval-floyd-$ver" -o "$log" \
    --wrap "bash $EVAL floyd $ver $flags"
  rc=$?
  if [ -f "$WORK/runs/$ver/summary.json" ]; then
    echo "[campaign] DONE $ver rc=$rc  $(grep -o 'GOODPUT.*' "$WORK/runs/$ver"/*.log 2>/dev/null | tail -1)"
  else
    echo "[campaign] WARN $ver rc=$rc — no summary.json (check log)"
  fi
done
echo "[campaign] ALL DONE  $(date -u)"
