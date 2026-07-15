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
WORK=$ROOT/workspace/sgl/v0.31/research/researchers/floyd
LOGD=$SGL/manager/.runtime/logs
NODE="${1:-slurm2-a3nodesetondem-3}"
EVAL="${2:-$SGL/researcher/.claude/skills/evaluation-sop/scripts/eval.sh}"
JOBSPEC="${3:-$WORK/analysis/jobs.txt}"   # file: "version<TAB or :>flags" per line

# version : extra eval flags  (stock has no flags). Same-node A/B, interleaved.
mapfile -t JOBS < <(grep -vE '^\s*(#|$)' "$JOBSPEC")

echo "[campaign] node=$NODE  $(date -u)"
for entry in "${JOBS[@]}"; do
  ver="${entry%%:*}"; flags="${entry#*:}"
  if [ -f "$WORK/runs/$ver/summary.json" ]; then
    echo "[campaign] SKIP $ver (summary.json exists)"; continue
  fi
  log="$LOGD/eval-floyd-$ver-%j.out"; mkdir -p "$LOGD" 2>/dev/null || log="/tmp/eval-floyd-$ver-%j.out"
  echo "[campaign] >>> $ver  flags='$flags'  $(date -u)"
  # shellcheck disable=SC2086
  # -t default 2:45:00: λ3,5 screens take ~2h; a hung job (NCCL flake) gets force-killed
  # here instead of stalling the serial driver for the full 6h wall limit. Set TIMEOUT
  # (e.g. 4:00:00) for a FULL λ{3,5,7,10} sweep, whose summary.json is written only after λ10.
  sbatch --wait -p a3 -N1 --exclusive --gres=gpu:8 -w "$NODE" -t "${TIMEOUT:-2:45:00}" \
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
