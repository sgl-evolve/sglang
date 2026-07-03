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

# Known persistently-small-disk pool node: skip by name (avoids a wasteful srun
# disk-check every poll). Override with SKIP_NODES env if it ever changes.
SKIP_NODES="${SKIP_NODES:-slurm2-a3nodeset0-0}"
held_nodes(){ for f in "$RT/held"/*; do [ -e "$f" ] && basename "$f"; done; }

attempts=0; MAX_ATTEMPTS=4   # retry transient init crashes (OOM race, node lottery)
while :; do
  any_free=0
  for node in $(held_nodes); do
    case " $SKIP_NODES " in *" $node "*) continue;; esac
    jid=$(cat "$RT/held/$node" 2>/dev/null) || continue
    squeue -h -j "$jid" >/dev/null 2>&1 || continue      # hold job dead -> skip
    exec 200>"$RT/locks/$node.lock"
    if flock -n 200; then
      any_free=1
      # disk gate + free-RAM gate under the lock. The RAM gate avoids the OOM
      # race where a node's flock frees before the previous eval's 768 GB pinned
      # host pool finishes releasing (new server then SIGKILLs during init).
      read free_kb mem_g < <(srun --jobid="$jid" --overlap -N1 -w "$node" bash -c \
        "echo \$(df --output=avail /mnt/localssd | tail -1) \$(free -g | awk '/^Mem:/{print \$7}')" 2>/dev/null)
      free_kb=${free_kb:-0}; mem_g=${mem_g:-0}
      if [[ "$free_kb" -ge 1932735283 && "$mem_g" -ge 1300 ]]; then   # 1.8 TiB disk, 1.3 TB RAM
        echo "[run_eval] node $node OK (${mem_g}G RAM, $((free_kb/1024/1024))G disk) -> running $VER"
        srun --jobid="$jid" --overlap -N1 -w "$node" --gres=gpu:8 bash "$EVAL" "$NAME" "$VER" "$@"
        rc=$?
        flock -u 200; exec 200>&-
        [ "$rc" -eq 0 ] && exit 0
        attempts=$((attempts+1))
        echo "[run_eval] eval on $node FAILED rc=$rc (attempt $attempts/$MAX_ATTEMPTS) -> retry on another node"
        [ "$attempts" -ge "$MAX_ATTEMPTS" ] && { echo "[run_eval] giving up after $attempts attempts"; exit "$rc"; }
        sleep 30   # let a crashed node's memory settle before re-checking it
      else
        echo "[run_eval] node $node not ready (${mem_g}G RAM, $((free_kb/1024/1024))G disk) -> skip"
        flock -u 200; exec 200>&-
      fi
    else
      exec 200>&-
    fi
  done
  echo "[run_eval] no free good-disk held node right now ($(date +%H:%M:%S)) — retry in 45s"
  sleep 15
done
