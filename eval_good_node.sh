#!/usr/bin/env bash
# eval_good_node.sh <name> <version> [eval args...]  — FINAL flock-first-hold design.
# Every 2s, try flock -n on each held pool node (instant local op, no probe latency). The moment a
# flock frees (previous eval-on-pool holder finished, incl. its L3 cleanup) we grab it — within 2s,
# beating competitors who re-poll every ~30s. Only AFTER holding the flock do we probe once (disk>=1.8T,
# no other sglang server, idle GPU); if usable, srun the frozen eval.sh; else release + short cooldown
# (covers bad-disk node 0-0 and non-flock bypassers whose server is still up). NEVER edits eval.sh.
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
echo "[good-node] $(date +%H:%M:%S) flock-first-hold racing for $VER"
declare -A cooldown; last_log=0
while :; do
  for n in $(held_nodes); do
    j=$(cat "$RT/held/$n" 2>/dev/null) || continue
    squeue -h -j "$j" >/dev/null 2>&1 || continue
    now=$(date +%s); [ "${cooldown[$n]:-0}" -gt "$now" ] && continue
    exec 200>"$RT/locks/$n.lock"
    if flock -n 200; then
      read -r fg ns gm <<<"$(srun --jobid="$j" --overlap -N1 -w "$n" bash -c '
        fg=$(df -BG /mnt/localssd 2>/dev/null|tail -1|awk "{gsub(/G/,\"\",\$4);print \$4}")
        ns=$(pgrep -c -f sglang.launch_server 2>/dev/null||echo 0)
        gm=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null|sort -n|tail -1)
        echo "${fg:-0} ${ns:-0} ${gm:-0}"' 2>/dev/null)"
      if [ "${ns:-99}" -eq 0 ] && [ "${gm:-99999}" -le 5000 ] && [ "${fg:-0}" -ge "$MIN_FREE_G" ]; then
        echo "[good-node] $(date +%H:%M:%S) WON $n (disk ${fg}G) — LAUNCH $VER"
        srun --jobid="$j" --overlap -N1 -w "$n" --gres=gpu:8 bash "$EVAL" "$NAME" "$VER" "$@"
        rc=$?; flock -u 200; exec 200>&-
        echo "[good-node] $(date +%H:%M:%S) eval exit rc=$rc on $n"; exit $rc
      fi
      flock -u 200; exec 200>&-
      cooldown[$n]=$(( $(date +%s) + 25 ))   # busy/bad-disk: brief backoff, keep racing others
    else
      exec 200>&-
    fi
  done
  now=$(date +%s); [ $((now-last_log)) -ge 120 ] && { echo "[good-node] $(date +%H:%M:%S) still racing..."; last_log=$now; }
  sleep 2
done
