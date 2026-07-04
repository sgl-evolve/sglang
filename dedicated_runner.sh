#!/usr/bin/env bash
# dedicated_runner.sh — sbatch BODY: runs on a dedicated --exclusive certified node (no flock race,
# no collision). Processes plan.tsv: for each not-done version run the frozen eval.sh on THIS node,
# retry once on crash, self-audit, log curated summary to W&B, mark done. If the FIRST version crashes
# both attempts before any success -> node is flaky -> exit 7 to release it (resubmit gets a fresh node).
# A later version crashing after a success is treated as version-specific (marked done, skipped). NEVER edits eval.sh.
set -uo pipefail
ROOT=/home/junyanch_google_com/autoresearch
set -a; . "$ROOT/.env"; set +a; export HF_TOKEN="${HF_API_KEY:-}"
NAME=kv-flint-2c
WS=$ROOT/workspace/sgl/researchers/$NAME
PROG=$ROOT/programs/sgl/researcher
EVAL=$PROG/.claude/skills/evaluation-sop/scripts/eval.sh
LOGW=$PROG/.claude/skills/report-sop/scripts/log_wandb.py
PLAN=$WS/plan.tsv; DONE=$WS/.done; mkdir -p "$DONE"; DLOG=$WS/dedicated.log
log(){ echo "[ded $(date '+%m-%d %H:%M:%S') $(hostname)] $*" >> "$DLOG"; }
audit(){ local out="$WS/runs/$1"
  [ -f "$out/summary.json" ] || { echo NO_SUMMARY; return; }
  [ -f "$out/resolved_args.json" ] || { echo NO_RESOLVED; return; }
  python3 - "$out/resolved_args.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1])); need={"context_length":"262144","mem_fraction_static":"0.85","hicache_size":"96","tp_size":"8"}
sys.exit(1 if [k for k,v in need.items() if str(r.get(k))!=v] else 0)
PY
  [ $? -ne 0 ] && { echo BUDGET_MISMATCH; return; }
  grep -q "SILENT FALLBACK" "$out/server.log" 2>/dev/null && { echo SILENT_FALLBACK; return; }
  echo PASS; }
ttft(){ python3 - "$WS/runs/$1/summary.json" <<'PY' 2>/dev/null
import json,sys
try: print(round(json.load(open(sys.argv[1]))["mix"]["ttft_mean_ms"],1))
except: print("?")
PY
}
log "==== dedicated runner START on $(hostname) (SLURM_JOB_ID=${SLURM_JOB_ID:-?}) ===="
node_ok=0
while IFS=$'\t' read -r ver tag env args _; do
  [ -z "${ver:-}" ] && continue; case "$ver" in \#*) continue;; esac
  [ -f "$DONE/$ver" ] && { log "skip $ver (done)"; continue; }
  mkdir -p "$WS/.claim"; exec 210>"$WS/.claim/$ver.lock"
  if ! flock -n 210; then log "skip $ver (claimed by another runner)"; exec 210>&-; continue; fi
  [ -f "$DONE/$ver" ] && { log "skip $ver (done after claim)"; flock -u 210; exec 210>&-; continue; }
  ENVKV=""; [ "$env" != "-" ] && ENVKV="$env"
  EARGS=(); [ "$args" != "-" ] && read -r -a EARGS <<<"$args"
  ok=0
  for attempt in 1 2; do
    log "== $ver attempt $attempt (env=$env args=$args) =="
    env $ENVKV bash "$EVAL" "$NAME" "$ver" ${EARGS[@]+"${EARGS[@]}"} < /dev/null >> "$DLOG" 2>&1
    rc=$?; log "$ver attempt $attempt rc=$rc"
    [ "$rc" = 0 ] && { ok=1; break; }
  done
  if [ "$ok" = 1 ]; then
    node_ok=1; a=$(audit "$ver"); t=$(ttft "$ver"); log "$ver audit=$a ttft_mean=${t}ms"
    if [ "$a" = PASS ]; then
      commit=$(git -C "$WS" rev-parse --short HEAD 2>/dev/null||echo '?')
      if ( cd "$WS"; source .venv/bin/activate; export PYTHONPATH="$WS/python"; python3 "$LOGW" "$NAME" "$WS/runs/$ver/summary.json" "$ver" "$commit" "$tag" ) >> "$DLOG" 2>&1; then
        log "$ver LOGGED to W&B [$tag] ttft_mean=${t}ms"
        printf '%s\t%s\t%s\tttft_mean=%sms\t%s\tlogged\n' "$(date '+%m-%d %H:%M')" "$ver" "$tag" "$t" "$commit" >> "$WS/results.tsv"
      else log "$ver W&B log FAILED"; fi
    else
      log "$ver audit failed=$a (not logged)"; printf '%s\t%s\t%s\t-\t-\taudit:%s\n' "$(date '+%m-%d %H:%M')" "$ver" "$tag" "$a" >> "$WS/results.tsv"
    fi
    touch "$DONE/$ver"
  else
    if [ "$node_ok" = 0 ]; then log "$ver failed 2x NO prior success -> BAD/FLAKY NODE $(hostname), record+exit"; echo "$(hostname)" >> "$WS/.bad_nodes"; exit 7; fi
    log "$ver failed 2x (version-specific; node was ok) -> mark done, skip"; touch "$DONE/$ver"
    printf '%s\t%s\t%s\t-\t-\tcrash2x\n' "$(date '+%m-%d %H:%M')" "$ver" "$tag" >> "$WS/results.tsv"
  fi
  flock -u 210 2>/dev/null; exec 210>&- 2>/dev/null
done < "$PLAN"
log "==== dedicated runner: plan exhausted ===="
