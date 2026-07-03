#!/usr/bin/env bash
# eval_good_node.sh <name> <version> [eval args...]  — FINAL launcher.
# Cheap sinfo FREE_MEM prefilter (login-node, no srun) rules out deep-busy nodes (a running eval pins
# host RAM). Only for a potentially-free node (high FREE_MEM) do we flock + srun-probe (disk>=1.8T, no
# other sglang server, idle GPU) and, if usable, srun the frozen eval.sh. 2s cadence to win the free
# window; collision-SAFE (never starts a 2nd server on an occupied node). NEVER edits eval.sh.
set -uo pipefail
ROOT=/home/junyanch_google_com/autoresearch
set -a; . "$ROOT/.env"; set +a
SGL_HOME="${SGL_HOME:-$ROOT/programs/sgl}"
EVAL="$SGL_HOME/researcher/.claude/skills/evaluation-sop/scripts/eval.sh"
RT="${SGL_RUNTIME:-$SGL_HOME/manager/.runtime}"
MIN_FREE_G=1800; MEM_PREFILTER_MB=600000   # skip srun-probe if FREE_MEM below this (deep-busy)
NAME="${1:?}"; VER="${2:?}"; shift 2
mkdir -p "$RT/locks" /tmp/kvflint_logs 2>/dev/null
held_nodes(){ compgen -G "$RT/held/*" >/dev/null 2>&1 && for f in "$RT/held"/*; do basename "$f"; done; }
echo "[good-node] $(date +%H:%M:%S) sinfo-prefilter racing for $VER"
last_log=0
while :; do
  nodes=$(held_nodes); [ -z "$nodes" ] && { echo "[good-node] no held nodes"; exit 1; }
  csv=$(echo "$nodes" | paste -sd,)
  # cheap: FREE_MEM per node, no srun
  declare -A freemem=()
  while read -r nn mm; do freemem[$nn]=${mm:-0}; done < <(sinfo -h -N -n "$csv" -o "%n %e" 2>/dev/null | sort -u)
  for n in $nodes; do
    [ "${freemem[$n]:-0}" -lt "$MEM_PREFILTER_MB" ] && continue    # deep-busy -> skip cheaply
    j=$(cat "$RT/held/$n" 2>/dev/null) || continue
    squeue -h -j "$j" >/dev/null 2>&1 || continue
    exec 200>"$RT/locks/$n.lock"
    if flock -n 200; then
      read -r fg ns gm <<<"$(srun --jobid="$j" --overlap -N1 -w "$n" bash -c '
        fg=$(df -BG /mnt/localssd 2>/dev/null|tail -1|awk "{gsub(/G/,\"\",\$4);print \$4}")
        ns=$(pgrep -c -f sglang.launch_server 2>/dev/null||echo 0)
        gm=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null|sort -n|tail -1)
        echo "${fg:-0} ${ns:-0} ${gm:-0}"' 2>/dev/null)"
      if [ "${ns:-99}" -eq 0 ] && [ "${gm:-99999}" -le 5000 ] && [ "${fg:-0}" -ge "$MIN_FREE_G" ]; then
        echo "[good-node] $(date +%H:%M:%S) WON $n (disk ${fg}G freemem ${freemem[$n]}MB) — LAUNCH $VER"
        srun --jobid="$j" --overlap -N1 -w "$n" --gres=gpu:8 bash "$EVAL" "$NAME" "$VER" "$@"
        rc=$?; flock -u 200; exec 200>&-
        echo "[good-node] $(date +%H:%M:%S) eval exit rc=$rc on $n"; exit $rc
      fi
      flock -u 200; exec 200>&-
    else exec 200>&-; fi
  done
  now=$(date +%s); [ $((now-last_log)) -ge 120 ] && { echo "[good-node] $(date +%H:%M:%S) still racing (freemem: $(for n in $nodes; do echo -n "$n=${freemem[$n]:-?} "; done))"; last_log=$now; }
  sleep 2
done
