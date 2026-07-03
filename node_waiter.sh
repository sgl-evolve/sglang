#!/usr/bin/env bash
# node_waiter.sh <node> [--dry] — kernel-queued (BLOCKING flock) on ONE held node, so I win the
# node the instant its foreign eval releases the flock (beats pollers). On a clean win, pop the next
# experiment from experiment_queue.txt and run it here, then autolog. Global cap: <=2 of my evals
# concurrent (good citizen — don't monopolize the pool). Run one instance per usable held node.
set -uo pipefail
NODE="$1"; DRY="${2:-}"
NAME=onyx-7q2
ROOT=/home/junyanch_google_com/autoresearch
EVAL=$ROOT/programs/sgl/researcher/.claude/skills/evaluation-sop/scripts/eval.sh
RT=$ROOT/programs/sgl/manager/.runtime
WORK=$ROOT/workspace/sgl/researchers/$NAME
Q=$WORK/experiment_queue.txt
cd "$WORK"
[ -f "$ROOT/.env" ] && { set -a; . "$ROOT/.env"; set +a; export HF_TOKEN="$HF_API_KEY"; export WANDB_API_KEY; }
case "$NODE" in *ondem-3) PORT=30751;; *nodeset-0) PORT=30752;; *nodeset1-2) PORT=30753;; *) PORT=30759;; esac

qpop(){ exec 201>"$Q.lock"; flock 201; local l; l=$(grep -vE '^[[:space:]]*$' "$Q" 2>/dev/null|head -1); [ -n "$l" ] && { grep -vFx "$l" "$Q">"$Q.tmp" 2>/dev/null; mv "$Q.tmp" "$Q"; }; flock -u 201; exec 201>&-; printf '%s' "$l"; }
myruns(){ pgrep -f "bash $EVAL $NAME v" 2>/dev/null | wc -l | tr -d ' '; }

while :; do
  [ -s "$Q" ] || { echo "[$NODE] queue empty -> exit"; exit 0; }
  jid=$(cat "$RT/held/$NODE" 2>/dev/null) || { sleep 20; continue; }
  squeue -h -j "$jid" >/dev/null 2>&1 || { sleep 20; continue; }
  exec 200>"$RT/locks/$NODE.lock"
  flock 200                                   # BLOCK until I acquire (kernel-queued)
  info=$(srun --jobid="$jid" --overlap -N1 -w "$NODE" bash -c "df -BG /mnt/localssd 2>/dev/null|tail -1|awk '{gsub(/G/,\"\",\$4);print \$4}'; free -g 2>/dev/null|awk '/Mem:/{print \$7}'; pgrep -c -f sglang.launch_server 2>/dev/null" 2>/dev/null)
  d=$(printf '%s\n' "$info"|sed -n 1p); m=$(printf '%s\n' "$info"|sed -n 2p); f=$(printf '%s\n' "$info"|sed -n 3p)
  if [ "${d:-0}" -lt 1800 ] || [ "${m:-0}" -lt 1400 ] || [ "${f:-0}" -gt 0 ]; then
    flock -u 200; exec 200>&-; sleep 8; continue      # node busy (foreign present) -> release, brief sleep
  fi
  # global citizen cap: at most 2 of my evals at once
  if [ "$(myruns)" -ge 2 ]; then flock -u 200; exec 200>&-; sleep 15; continue; fi
  exp=$(qpop); [ -z "$exp" ] && { flock -u 200; exec 200>&-; exit 0; }
  IFS='|' read -r lbl env flags tag <<<"$exp"
  if [ "$DRY" = "--dry" ]; then echo "[$NODE] DRY would run $lbl env=[$env] flags=[$flags]"; flock -u 200; exec 200>&-; exit 0; fi
  echo "[$NODE] WON -> $lbl (port $PORT)"
  rm -rf "$WORK/runs/$lbl" 2>/dev/null
  ( unset SGLANG_HICACHE_FILE_READ_THREADS SGLANG_HICACHE_PREFETCH_TIMEOUT_BASE SGLANG_HICACHE_PREFETCH_TIMEOUT_PER_KI SGLANG_HICACHE_PREFETCH_TIMEOUT_MAX
    [ "$env" != "-" ] && export $env
    export PORT
    srun --jobid="$jid" --overlap -N1 -w "$NODE" --gres=gpu:8 --export=ALL,PORT="$PORT" bash "$EVAL" "$NAME" "$lbl" --enforce-disable-flashinfer-allreduce-fusion $flags > "eval-$lbl.log" 2>&1 )
  echo "[$NODE] $lbl finished rc=$?"
  [ -f "runs/$lbl/summary.json" ] && { bash finish_eval.sh "$lbl" "$tag" >> "logs_finish_$lbl.txt" 2>&1; touch "runs/$lbl/.logged"; }
  flock -u 200; exec 200>&-
done
