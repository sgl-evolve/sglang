#!/usr/bin/env bash
# keeper.sh — keeps exactly one dedicated_runner sbatch (ded-kvflint) queued/running until plan.tsv is
# fully done. Resubmits after a flaky-node exit or completion. Detached (setsid) -> survives teardowns.
set -uo pipefail
ROOT=/home/junyanch_google_com/autoresearch
set -a; . "$ROOT/.env"; set +a
WS=$ROOT/workspace/sgl/researchers/kv-flint-2c
RUNNER=$WS/dedicated_runner.sh; PLAN=$WS/plan.tsv; DONE=$WS/.done; mkdir -p "$DONE" /tmp/kvflint_logs
BASEEXCL="slurm2-a3nodeset-2,slurm2-a3nodeset1-0,slurm2-a3nodeset1-1,slurm2-a3nodesetondem-0,slurm2-a3nodesetondem-1,slurm2-a3nodesetondem-2"
exec 8>/tmp/kvflint_logs/keeper.lock; flock -n 8 || { echo "keeper already running"; exit 0; }
klog(){ echo "[keep $(date '+%m-%d %H:%M:%S')] $*" >> "$WS/keeper.log"; }
me=$(whoami)
klog "==== keeper START (pid $$) ===="
while :; do
  alldone=1
  while IFS=$'\t' read -r ver _; do [ -z "${ver:-}" ] && continue; case "$ver" in \#*) continue;; esac; [ -f "$DONE/$ver" ] || alldone=0; done < "$PLAN"
  [ "$alldone" = 1 ] && { klog "plan fully done -> keeper exit"; break; }
  n=$(squeue -u "$me" -h -o "%j" 2>/dev/null | grep -c "ded-kvflint" || true)
  if [ "${n:-0}" = 0 ]; then
    BAD=$(sort -u "$WS/.bad_nodes" 2>/dev/null | paste -sd, ); EXCL="$BASEEXCL${BAD:+,$BAD}"; jid=$(sbatch --parsable -p a3 -N1 --exclusive --gres=gpu:8 --exclude="$EXCL" -t 12:00:00 \
          -J ded-kvflint -o /tmp/kvflint_logs/ded-%j.out "$RUNNER" 2>>"$WS/keeper.log") \
      && klog "submitted ded-kvflint job $jid" || klog "sbatch submit failed"
  fi
  sleep 120
done
