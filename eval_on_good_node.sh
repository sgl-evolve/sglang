#!/usr/bin/env bash
# Same mechanism as eval-on-pool.sh (flock a held node, srun the frozen eval.sh in),
# but skips held nodes whose /mnt/localssd has < 1.8T free. Fairness: we acquire the
# node's flock FIRST (cheap, no srun), and only then srun a df-check — so we never run
# anything on a node that is busy with another researcher's --exclusive eval.
set -uo pipefail
ROOT=/home/junyanch_google_com/autoresearch
set -a; . "$ROOT/.env"; set +a
SGL_HOME="${SGL_HOME:-$ROOT/programs/sgl}"
EVAL="$SGL_HOME/researcher/.claude/skills/evaluation-sop/scripts/eval.sh"
RT="${SGL_RUNTIME:-$SGL_HOME/manager/.runtime}"
NAME="$1"; VER="$2"; shift 2
held(){ for f in "$RT/held"/*; do basename "$f"; done; }
while :; do
  for node in $(held); do
    jid=$(cat "$RT/held/$node" 2>/dev/null) || continue
    squeue -h -j "$jid" >/dev/null 2>&1 || continue        # hold job dead -> skip
    exec 200>"$RT/locks/$node.lock"
    if flock -n 200; then                                   # got the node (idle) -> now safe to probe
      free=$(timeout 40 srun --jobid="$jid" --overlap -N1 -w "$node" bash -c \
          "df --output=avail -BG /mnt/localssd | tail -1 | tr -dc 0-9" 2>/dev/null)
      if [ -n "$free" ] && [ "$free" -ge 1800 ]; then
        echo "[good-node] running eval $VER on held node $node (${free}G free, hold $jid)"
        srun --jobid="$jid" --overlap -N1 -w "$node" --gres=gpu:8 bash "$EVAL" "$NAME" "$VER" "$@"
        rc=$?; flock -u 200; exec 200>&-; exit $rc
      fi
      echo "[good-node] locked $node but only ${free:-?}G free (<1800) -> release, skip"
      flock -u 200; exec 200>&-
    else
      exec 200>&-                                           # busy with another researcher's eval
    fi
  done
  echo "[good-node] no free held node with >=1.8T right now — wait 45s"; sleep 45
done
