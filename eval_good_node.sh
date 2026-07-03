#!/usr/bin/env bash
# eval_good_node.sh <name> <version> [eval args...]
# Race-winning, collision-SAFE eval launcher on the manager's held pool.
# The pool flock is NOT authoritative (some researchers bypass it with their own launchers), so the
# reliable readiness signal is: node has NO sglang.launch_server AND disk>=1.8T free AND idle GPUs.
# We parallel-probe ALL held nodes every ~5s (fast detection to win the "who starts a server first
# after a node frees" race), and the instant one is ready we flock (coordinate with flock-users) +
# re-probe + srun the frozen eval.sh into it. NEVER edits eval.sh.
set -uo pipefail
ROOT=/home/junyanch_google_com/autoresearch
set -a; . "$ROOT/.env"; set +a
SGL_HOME="${SGL_HOME:-$ROOT/programs/sgl}"
EVAL="$SGL_HOME/researcher/.claude/skills/evaluation-sop/scripts/eval.sh"
RT="${SGL_RUNTIME:-$SGL_HOME/manager/.runtime}"
MIN_FREE_G=1800
NAME="${1:?}"; VER="${2:?}"; shift 2
mkdir -p "$RT/locks" /tmp/kvflint_logs 2>/dev/null
held_nodes(){ compgen -G "$RT/held/*" >/dev/null 2>&1 && for f in "$RT/held"/*; do basename "$f"; done; }
probe(){ srun --jobid="$2" --overlap -N1 -w "$1" bash -c '
    fg=$(df -BG /mnt/localssd 2>/dev/null|tail -1|awk "{gsub(/G/,\"\",\$4);print \$4}")
    ns=$(pgrep -c -f sglang.launch_server 2>/dev/null||echo 0)
    gm=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null|sort -n|tail -1)
    echo "${fg:-0} ${ns:-0} ${gm:-0}"' 2>/dev/null; }
echo "[good-node] $(date +%H:%M:%S) racing (parallel-probe 5s) for a held node for $VER"
last_log=0
while :; do
  tmp=$(mktemp); alive=0
  for n in $(held_nodes); do
    j=$(cat "$RT/held/$n" 2>/dev/null) || continue
    squeue -h -j "$j" >/dev/null 2>&1 || continue
    alive=1
    ( echo "$n $j $(probe "$n" "$j")" >> "$tmp" ) &
  done
  wait
  [ "$alive" = 0 ] && { echo "[good-node] no live held nodes"; rm -f "$tmp"; exit 1; }
  while read -r n j fg ns gm; do
    [ "${ns:-99}" -ne 0 ] && continue
    [ "${gm:-99999}" -gt 5000 ] && continue
    [ "${fg:-0}" -lt "$MIN_FREE_G" ] && continue
    exec 200>"$RT/locks/$n.lock"
    if flock -n 200; then
      read -r fg2 ns2 gm2 <<<"$(probe "$n" "$j")"
      if [ "${ns2:-99}" -eq 0 ] && [ "${gm2:-99999}" -le 5000 ] && [ "${fg2:-0}" -ge "$MIN_FREE_G" ]; then
        echo "[good-node] $(date +%H:%M:%S) WON $n (disk ${fg2}G) — LAUNCH $VER"
        srun --jobid="$j" --overlap -N1 -w "$n" --gres=gpu:8 bash "$EVAL" "$NAME" "$VER" "$@"
        rc=$?; flock -u 200; exec 200>&-; rm -f "$tmp"
        echo "[good-node] $(date +%H:%M:%S) eval exit rc=$rc on $n"; exit $rc
      fi
      flock -u 200; exec 200>&-
    else exec 200>&-; fi
  done < <(sort -k3 -nr "$tmp")
  rm -f "$tmp"
  now=$(date +%s); [ $((now-last_log)) -ge 120 ] && { echo "[good-node] $(date +%H:%M:%S) still racing..."; last_log=$now; }
  sleep 5
done
