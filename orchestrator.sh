#!/usr/bin/env bash
# orchestrator.sh — autonomous research loop for kv-flint-2c (survives session teardowns via setsid).
# For each version in plan.tsv (TAB: version<TAB>tag<TAB>envKV|-<TAB>evalargs|-):
#   1) acquire a held-pool node + run the frozen eval.sh (eval_good_node.sh, env-injected),
#   2) self-audit (resolved_args match contract, no SILENT FALLBACK, eval rc=0),
#   3) if pass, log curated summary.json to W&B (report SOP) and append to report.md,
#   4) mark done, advance. Single-instance (flock). Resumable (done marks). NEVER edits eval.sh.
set -uo pipefail
ROOT=/home/junyanch_google_com/autoresearch
set -a; . "$ROOT/.env"; set +a
export HF_TOKEN="${HF_API_KEY:-}"
NAME=kv-flint-2c
WS=$ROOT/workspace/sgl/researchers/$NAME
PROG=$ROOT/programs/sgl/researcher
LOGW=$PROG/.claude/skills/report-sop/scripts/log_wandb.py
LAUNCH=$WS/eval_good_node.sh
PLAN=$WS/plan.tsv
OLOG=/tmp/kvflint_logs/orchestrator.log
DONEDIR=/tmp/kvflint_logs/orch_done; mkdir -p "$DONEDIR" /tmp/kvflint_logs
export NBARGE="${NBARGE:-2}"
exec 9>/tmp/kvflint_logs/orch.lock
flock -n 9 || { echo "orchestrator already running; exit"; exit 0; }
log(){ echo "[orch $(date '+%m-%d %H:%M:%S')] $*" >> "$OLOG"; }

audit(){ # $1=version ; echo PASS or a reason
  local ver="$1" out="$WS/runs/$1"
  [ -f "$out/summary.json" ] || { echo NO_SUMMARY; return; }
  [ -f "$out/resolved_args.json" ] || { echo NO_RESOLVED; return; }
  python3 - "$out/resolved_args.json" <<'PY' 2>/dev/null
import json,sys
r=json.load(open(sys.argv[1]))
need={"context_length":"262144","mem_fraction_static":"0.85","hicache_size":"96","tp_size":"8"}
sys.exit(1 if [k for k,v in need.items() if str(r.get(k))!=v] else 0)
PY
  [ $? -ne 0 ] && { echo BUDGET_MISMATCH; return; }
  grep -q "SILENT FALLBACK" "$out/server.log" 2>/dev/null && { echo SILENT_FALLBACK; return; }
  echo PASS
}

ttft(){ python3 - "$WS/runs/$1/summary.json" <<'PY' 2>/dev/null
import json,sys
try: print(round(json.load(open(sys.argv[1]))["mix"]["ttft_mean_ms"],1))
except: print("?")
PY
}

log "==== orchestrator START (pid $$) NBARGE=$NBARGE ===="
while IFS=$'\t' read -r ver tag env args _rest; do
  [ -z "${ver:-}" ] && continue
  case "$ver" in \#*) continue;; esac
  if [ -f "$DONEDIR/$ver" ]; then log "skip $ver (already done)"; continue; fi
  log "==== VERSION $ver  tag=$tag  env=$env  args=$args ===="
  ENVKV=""; [ "$env" != "-" ] && ENVKV="$env"
  EARGS=(); [ "$args" != "-" ] && read -r -a EARGS <<<"$args"
  rm -f "/tmp/kvflint_logs/${NAME}_${ver}_state/rc" 2>/dev/null
  # acquire node + run eval (blocks: win race + ~2.5h eval)
  env $ENVKV bash "$LAUNCH" "$NAME" "$ver" ${EARGS[@]+"${EARGS[@]}"} < /dev/null >> "$OLOG" 2>&1
  rc=$(cat "/tmp/kvflint_logs/${NAME}_${ver}_state/rc" 2>/dev/null || echo 1)
  log "$ver eval finished rc=$rc"
  if [ "$rc" = 0 ]; then
    a=$(audit "$ver"); t=$(ttft "$ver"); log "$ver audit=$a ttft_mean=${t}ms"
    if [ "$a" = PASS ]; then
      commit=$(git -C "$WS" rev-parse --short HEAD 2>/dev/null || echo '?')
      if ( cd "$WS"; source .venv/bin/activate; export PYTHONPATH="$WS/python"; python3 "$LOGW" "$NAME" "$WS/runs/$ver/summary.json" "$ver" "$commit" "$tag" ) >> "$OLOG" 2>&1; then
        log "$ver LOGGED to W&B [$tag] ttft_mean=${t}ms commit=$commit"
        printf '%s\t%s\t%s\tttft_mean=%sms\tcommit=%s\t%s\n' "$(date '+%m-%d %H:%M')" "$ver" "$tag" "$t" "$commit" "logged" >> "$WS/results.tsv"
      else
        log "$ver W&B log FAILED (see $OLOG)"
      fi
    else
      log "$ver NOT logged (audit=$a)"
      printf '%s\t%s\t%s\t-\t-\t%s\n' "$(date '+%m-%d %H:%M')" "$ver" "$tag" "audit:$a" >> "$WS/results.tsv"
    fi
  else
    log "$ver FAILED rc=$rc (not logged)"
    printf '%s\t%s\t%s\t-\t-\trc=%s\n' "$(date '+%m-%d %H:%M')" "$ver" "$tag" "$rc" >> "$WS/results.tsv"
  fi
  touch "$DONEDIR/$ver"
done < "$PLAN"
log "==== plan exhausted; orchestrator idle (append to plan.tsv + relaunch to continue) ===="
