#!/usr/bin/env bash
# eval_good_node.sh <name> <version> [eval args...]  — tight barge-poll launcher.
# Linux flock granting is NOT FIFO: a fresh `flock -n` at the release instant BARGES ahead of already-
# blocked waiters. So competitors win by tight-polling `flock -n`. We match: one background barger per
# held pool node, each attempting `flock -n` every ~0.2s (cheap local op) to grab a node's lock the
# instant it frees. On winning, probe (disk>=1.8T, no other sglang server, idle GPU); if usable, claim a
# single global run-lock and srun the frozen eval.sh (holding the lock for the run); winner kills siblings.
# Collision-SAFE; NEVER edits eval.sh.
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
    ns=$(pgrep -c -f sglang.launch_server 2>/dev/null||echo 0)
    gm=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null|sort -n|tail -1)
    echo "${fg:-0} ${ns:-0} ${gm:-0}"' 2>/dev/null; }

barger(){
  local n="$1" j="$2"; shift 2
  local next_sq=0
  while [ ! -f "$STATE/winner" ]; do
    # cheap: only re-check hold-job liveness every ~20s
    now=$(date +%s)
    if [ "$now" -ge "$next_sq" ]; then squeue -h -j "$j" >/dev/null 2>&1 || { sleep 15; next_sq=$((now+20)); continue; }; next_sq=$((now+20)); fi
    exec 200>"$RT/locks/$n.lock"
    if flock -n 200; then
      # won the flock (node's pool slot is free). Probe once.
      if [ ! -f "$STATE/winner" ]; then
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
      fi
      flock -u 200; exec 200>&-
      sleep 10          # won flock but node not usable (bad disk / busy) -> back off, don't hammer
    else
      exec 200>&-
      sleep 0.2         # flock held by a running eval -> tight poll to catch its release window
    fi
  done
}

echo "[good-node] $(date +%H:%M:%S) tight barge-poll racing for $VER on: $(held_nodes|tr '\n' ' ')"
: > "$STATE/pids"; pids=()
NBARGE="${NBARGE:-3}"
for n in $(held_nodes); do
  j=$(cat "$RT/held/$n" 2>/dev/null) || continue
  for k in $(seq 1 "$NBARGE"); do barger "$n" "$j" "$@" & p=$!; pids+=($p); echo "$p" >> "$STATE/pids"; done
done
last=0
while [ ! -f "$STATE/rc" ]; do
  alive=0; for p in "${pids[@]}"; do kill -0 "$p" 2>/dev/null && { alive=1; break; }; done
  [ "$alive" = 0 ] && [ ! -f "$STATE/winner" ] && { echo "[good-node] all bargers died, respawn"; exec "$0" "$NAME" "$VER" "$@"; }
  now=$(date +%s); [ $((now-last)) -ge 180 ] && { echo "[good-node] $(date +%H:%M:%S) still barge-polling..."; last=$now; }
  sleep 15
done
wait 2>/dev/null
echo "[good-node] $(date +%H:%M:%S) launcher exit rc=$(cat "$STATE/rc" 2>/dev/null||echo 1)"
exit "$(cat "$STATE/rc" 2>/dev/null||echo 1)"
