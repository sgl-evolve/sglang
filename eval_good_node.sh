#!/usr/bin/env bash
# eval_good_node.sh <name> <version> [eval args...]
# Collision-SAFE eval launcher on the manager's held pool. Only launches on a held node that is:
#   (1) disk >= 1.8T free on /mnt/localssd,
#   (2) NOT already running another sglang.launch_server (avoid OOM-killing a neighbor -> fairness),
#   (3) GPUs essentially idle (nvidia-smi),
#   (4) acquirable via the per-node flock (coordinate with other eval-on-pool users).
# Waits (loops) until such a node appears. NEVER edits the frozen eval.sh.
set -uo pipefail
ROOT=/home/junyanch_google_com/autoresearch
set -a; . "$ROOT/.env"; set +a
SGL_HOME="${SGL_HOME:-$ROOT/programs/sgl}"
EVAL="$SGL_HOME/researcher/.claude/skills/evaluation-sop/scripts/eval.sh"
RT="${SGL_RUNTIME:-$SGL_HOME/manager/.runtime}"
MIN_FREE_G=1800
NAME="${1:?usage: eval_good_node.sh <name> <version> [eval args...]}"
VER="${2:?}"; shift 2

held_nodes(){ compgen -G "$RT/held/*" >/dev/null 2>&1 && for f in "$RT/held"/*; do basename "$f"; done; }
mkdir -p "$RT/locks" 2>/dev/null

probe(){ # runs a probe on the node; prints "FREE_G;NSERVERS;GPUMEM_MIB"
  srun --jobid="$1" --overlap -N1 -w "$2" bash -c '
    fg=$(df -BG /mnt/localssd 2>/dev/null | tail -1 | awk "{gsub(/G/,\"\",\$4);print \$4}")
    ns=$(pgrep -c -f "sglang.launch_server" 2>/dev/null || echo 0)
    gm=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | sort -n | tail -1)
    echo "${fg:-0};${ns:-0};${gm:-0}"
  ' 2>/dev/null
}

echo "[good-node] $(date +%H:%M:%S) looking for a clean held node for $VER ..."
while :; do
  any_alive=0
  for node in $(held_nodes); do
    jid=$(cat "$RT/held/$node" 2>/dev/null) || continue
    squeue -h -j "$jid" >/dev/null 2>&1 || continue
    any_alive=1
    read -r fg ns gm <<<"$(probe "$jid" "$node" | tr ';' ' ')"
    if [ "${fg:-0}" -lt "$MIN_FREE_G" ]; then echo "[good-node] skip $node: disk ${fg}G<${MIN_FREE_G}G"; continue; fi
    if [ "${ns:-0}" -gt 0 ]; then echo "[good-node] skip $node: $ns sglang server(s) already running (neighbor busy)"; continue; fi
    if [ "${gm:-0}" -gt 5000 ]; then echo "[good-node] skip $node: GPU busy (${gm} MiB used)"; continue; fi
    exec 200>"$RT/locks/$node.lock"
    if flock -n 200; then
      # re-probe under the lock to close the TOCTOU window
      read -r fg2 ns2 gm2 <<<"$(probe "$jid" "$node" | tr ';' ' ')"
      if [ "${ns2:-0}" -gt 0 ] || [ "${gm2:-0}" -gt 5000 ]; then
        echo "[good-node] $node became busy under lock (ns=$ns2 gpu=${gm2}MiB) - releasing"; flock -u 200; exec 200>&-; continue
      fi
      echo "[good-node] $(date +%H:%M:%S) LAUNCH $VER on $node (job $jid, disk ${fg2}G, gpu ${gm2}MiB)"
      srun --jobid="$jid" --overlap -N1 -w "$node" --gres=gpu:8 bash "$EVAL" "$NAME" "$VER" "$@"
      rc=$?
      flock -u 200; exec 200>&-
      echo "[good-node] $(date +%H:%M:%S) eval exit rc=$rc on $node"
      exit $rc
    fi
    exec 200>&-
  done
  [ "$any_alive" -eq 0 ] && { echo "[good-node] no live held nodes"; exit 1; }
  sleep 45
done
