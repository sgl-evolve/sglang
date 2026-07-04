#!/usr/bin/env bash
# eval_good_node.sh <name> <version> [eval args...]  — INFINITE-BLOCKING flock launcher.
# Competitors use `flock -w` (blocking): the kernel grants a freed lock to a blocked waiter instantly,
# beating any `flock -n` poll. So we BLOCK (no timeout) to sit in the kernel wake queue and hold FIFO
# position (never reset by re-blocking). One+ background waiter per held pool node (NBARGE per node).
# On acquiring: verify usable (disk>=1.8T, no other sglang server, idle GPU); if usable claim one global
# run-lock and srun the frozen eval.sh (holding the lock for the run); winner kills siblings. A node that
# is not usable (bad-disk 0-0 / rare mid-cleanup) gets a 10s backoff before re-blocking. NEVER edits eval.sh.
set -uo pipefail
ROOT=/home/junyanch_google_com/autoresearch
set -a; . "$ROOT/.env"; set +a
SGL_HOME="${SGL_HOME:-$ROOT/programs/sgl}"
EVAL="$SGL_HOME/researcher/.claude/skills/evaluation-sop/scripts/eval.sh"
RT="${SGL_RUNTIME:-$SGL_HOME/manager/.runtime}"
MIN_FREE_G=1800
NAME="${1:?}"; VER="${2:?}"; shift 2
STATE="/tmp/kvflint_logs/${NAME}_${VER}_state"; mkdir -p "$STATE"; rm -f "$STATE/winner" "$STATE/rc" "$STATE/pids"
GLOCK="$STATE/global.lock"
held_nodes(){ compgen -G "$RT/held/*" >/dev/null 2>&1 && for f in "$RT/held"/*; do basename "$f"; done; }
probe(){ srun --jobid="$2" --overlap -N1 -w "$1" bash -c '
    fg=$(df -BG /mnt/localssd 2>/dev/null|tail -1|awk "{gsub(/G/,\"\",\$4);print \$4}")
    ns=$(pgrep -c -f "[s]glang.launch_server" 2>/dev/null||echo 0)
    gm=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null|sort -n|tail -1)
    echo "${fg:-0} ${ns:-0} ${gm:-0}"' 2>/dev/null; }

waiter(){
  local n="$1" j="$2"; shift 2
  while [ ! -f "$STATE/winner" ]; do
    exec 200>"$RT/locks/$n.lock"
    flock 200                      # BLOCK indefinitely -> hold FIFO wake-queue position
    if [ -f "$STATE/winner" ]; then flock -u 200; exec 200>&-; return 0; fi
    if ! squeue -h -j "$j" >/dev/null 2>&1; then flock -u 200; exec 200>&-; sleep 30; continue; fi
    read -r fg ns gm <<<"$(probe "$n" "$j")"
    if [ "${ns:-99}" -eq 0 ] && [ "${gm:-99999}" -le 5000 ] && [ "${fg:-0}" -ge "$MIN_FREE_G" ]; then
      exec 201>"$GLOCK"
      if flock -n 201 && [ ! -f "$STATE/winner" ]; then
        echo "$n" > "$STATE/winner"
        for sp in $(cat "$STATE/pids" 2>/dev/null); do [ "$sp" != "$BASHPID" ] && kill "$sp" 2>/dev/null; done
        echo "[good-node] $(date +%H:%M:%S) WON $n (disk ${fg}G) — LAUNCH $VER"
        srun --jobid="$j" --overlap -N1 -w "$n" --gres=gpu:8 bash "$EVAL" "$NAME" "$VER" "$@"
        echo "$?" > "$STATE/rc"; flock -u 201; flock -u 200
        echo "[good-node] $(date +%H:%M:%S) eval done on $n (rc=$(cat "$STATE/rc"))"; return 0
      fi
      flock -u 201 2>/dev/null; exec 201>&-
    fi
    flock -u 200; exec 200>&-
    sleep 10                       # acquired but not usable -> back off before re-blocking
  done
}

echo "[good-node] $(date +%H:%M:%S) INFINITE-BLOCKING racing for $VER on: $(held_nodes|tr '\n' ' ')"
: > "$STATE/pids"; pids=(); NBARGE="${NBARGE:-3}"
for n in $(held_nodes); do
  j=$(cat "$RT/held/$n" 2>/dev/null) || continue
  for k in $(seq 1 "$NBARGE"); do waiter "$n" "$j" "$@" & p=$!; pids+=($p); echo "$p" >> "$STATE/pids"; done
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
