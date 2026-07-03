#!/usr/bin/env bash
# run_eval.sh <version> [eval policy args...]
# Disk-aware wrapper around the manager's held eval pool: grabs a free held CERTIFIED node
# behind the SAME per-node flock eval-on-pool uses (so we never collide with other researchers),
# but only accepts a node with >=1.8T free /mnt/localssd (skips the pool's small-disk node).
# Runs the FROZEN eval.sh into it via srun --overlap. Blocks ~2h -> run in background.
set -uo pipefail
ROOT=/home/junyanch_google_com/autoresearch
[ -f "$ROOT/.env" ] && { set -a; . "$ROOT/.env"; set +a; export HF_TOKEN="${HF_API_KEY:-}"; }
NAME=kv-lynx-4d2
RT=/home/junyanch_google_com/autoresearch/programs/sgl/manager/.runtime
EVAL=$ROOT/programs/sgl/researcher/.claude/skills/evaluation-sop/scripts/eval.sh
VER="${1:?usage: run_eval.sh <version> [args...]}"; shift || true

held_nodes(){ for f in "$RT/held"/*; do [ -e "$f" ] && basename "$f"; done; }

while :; do
  any_free=0
  for node in $(held_nodes); do
    jid=$(cat "$RT/held/$node" 2>/dev/null) || continue
    squeue -h -j "$jid" >/dev/null 2>&1 || continue      # hold job dead -> skip
    exec 200>"$RT/locks/$node.lock"
    if flock -n 200; then
      any_free=1
      # disk gate under the lock
      free_kb=$(srun --jobid="$jid" --overlap -N1 -w "$node" bash -c "df --output=avail /mnt/localssd | tail -1" 2>/dev/null | tr -d ' ')
      if [[ -n "$free_kb" && "$free_kb" -ge 1932735283 ]]; then   # 1.8 TiB in KB
        echo "[run_eval] node $node OK ($((free_kb/1024/1024))G free) -> running $VER"
        srun --jobid="$jid" --overlap -N1 -w "$node" --gres=gpu:8 bash "$EVAL" "$NAME" "$VER" "$@"
        rc=$?
        flock -u 200; exec 200>&-
        exit $rc
      else
        echo "[run_eval] node $node disk too small ($((free_kb/1024/1024))G) -> skip"
        flock -u 200; exec 200>&-
      fi
    else
      exec 200>&-
    fi
  done
  echo "[run_eval] no free good-disk held node right now ($(date +%H:%M:%S)) — retry in 45s"
  sleep 45
done
