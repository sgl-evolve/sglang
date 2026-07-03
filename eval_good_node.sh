#!/usr/bin/env bash
# eval_good_node.sh <name> <version> [eval args...]  — continuous-blocking-flock launcher.
# CRITICAL: Linux flock wakes blocked waiters ~FIFO. A short `flock -w` timeout re-queues you at the
# BACK every timeout, so continuously-blocked competitors always win. Fix: one background waiter per
# held pool node, each blocking CONTINUOUSLY (long `flock -w`) to hold its FIFO position. On acquiring
# (node freed), probe (disk>=1.8T, no other sglang server, idle GPU); if usable, claim a single global
# run-lock and srun the frozen eval.sh (holding the flock for the run); the winner kills the siblings.
# Collision-SAFE; NEVER edits eval.sh.
set -uo pipefail
ROOT=/home/junyanch_google_com/autoresearch
set -a; . "$ROOT/.env"; set +a
SGL_HOME="${SGL_HOME:-$ROOT/programs/sgl}"
EVAL="$SGL_HOME/researcher/.claude/skills/evaluation-sop/scripts/eval.sh"
RT="${SGL_RUNTIME:-$SGL_HOME/manager/.runtime}"
MIN_FREE_G=1800; BLOCK=7200
NAME="${1:?}"; VER="${2:?}"; shift 2
STATE="/tmp/kvflint_logs/${NAME}_${VER}_state"; mkdir -p "$STATE"; rm -f "$STATE/winner" "$STATE/rc" "$STATE/pids"
GLOCK="$STATE/global.lock"
held_nodes(){ compgen -G "$RT/held/*" >/dev/null 2>&1 && for f in "$RT/held"/*; do basename "$f"; done; }
probe(){ srun --jobid="$2" --overlap -N1 -w "$1" bash -c '
    fg=$(df -BG /mnt/localssd 2>/dev/null|tail -1|awk "{gsub(/G/,\"\",\$4);print \$4}")
    ns=$(pgrep -c -f sglang.launch_server 2>/dev/null||echo 0)
    gm=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null|sort -n|tail -1)
    echo "${fg:-0} ${ns:-0} ${gm:-0}"' 2>/dev/null; }

waiter(){
  local n="$1" j="$2"; shift 2
  while [ ! -f "$STATE/winner" ]; do
    squeue -h -j "$j" >/dev/null 2>&1 || { sleep 20; continue; }
    exec 200>"$RT/locks/$n.lock"
    if flock -w "$BLOCK" 200; then                 # continuous block -> hold FIFO position
      [ -f "$STATE/winner" ] && { flock -u 200; exec 200>&-; return 0; }
      read -r fg ns gm <<<"$(probe "$n" "$j")"
      if [ "${ns:-99}" -eq 0 ] && [ "${gm:-99999}" -le 5000 ] && [ "${fg:-0}" -ge "$MIN_FREE_G" ]; then
        exec 201>"$GLOCK"
        if flock -n 201 && [ ! -f "$STATE/winner" ]; then
          echo "$n" > "$STATE/winner"
          # kill sibling waiters (all pids in state file except me)
          for sp in $(cat "$STATE/pids" 2>/dev/null); do [ "$sp" != "$BASHPID" ] && kill "$sp" 2>/dev/null; done
          echo "[good-node] $(date +%H:%M:%S) WON $n (disk ${fg}G) — LAUNCH $VER"
          srun --jobid="$j" --overlap -N1 -w "$n" --gres=gpu:8 bash "$EVAL" "$NAME" "$VER" "$@"
          echo "$?" > "$STATE/rc"
          flock -u 201; flock -u 200
          echo "[good-node] $(date +%H:%M:%S) eval done on $n (rc=$(cat "$STATE/rc"))"
          return 0
        fi
        flock -u 201 2>/dev/null; exec 201>&-
      fi
      flock -u 200; exec 200>&-
      sleep 3                                        # transient reject (mid-cleanup/bad disk) -> re-block
    fi
  done
}

echo "[good-node] $(date +%H:%M:%S) continuous-blocking racing for $VER on: $(held_nodes|tr '\n' ' ')"
: > "$STATE/pids"
pids=()
for n in $(held_nodes); do
  j=$(cat "$RT/held/$n" 2>/dev/null) || continue
  waiter "$n" "$j" "$@" &
  p=$!; pids+=($p); echo "$p" >> "$STATE/pids"
done
last=0
while [ ! -f "$STATE/rc" ]; do
  alive=0; for p in "${pids[@]}"; do kill -0 "$p" 2>/dev/null && { alive=1; break; }; done
  [ "$alive" = 0 ] && [ ! -f "$STATE/winner" ] && { echo "[good-node] all waiters died, respawn"; exec "$0" "$NAME" "$VER" "$@"; }
  now=$(date +%s); [ $((now-last)) -ge 180 ] && { echo "[good-node] $(date +%H:%M:%S) still blocking on all nodes..."; last=$now; }
  sleep 15
done
wait 2>/dev/null
echo "[good-node] $(date +%H:%M:%S) launcher exit rc=$(cat "$STATE/rc" 2>/dev/null||echo 1)"
exit "$(cat "$STATE/rc" 2>/dev/null||echo 1)"
