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
      # disk + free-RAM + GPU-idle gate under the lock. CRUCIAL: some researchers
      # run evals on these held nodes WITHOUT the pool flock (e.g. pinned-eval),
      # so a flock-free node can still have a neighbor's 8-GPU server on it. The
      # GPU-idle check (max GPU mem < 10 GB) ensures the node is TRULY idle before
      # we launch, else our server collides -> OOM/NCCL-hang/SIGKILL. The RAM gate
      # also avoids the teardown race on a freshly-freed node.
      read free_kb mem_g gpu_max < <(srun --jobid="$jid" --overlap -N1 -w "$node" bash -c \
        "echo \$(df --output=avail /mnt/localssd | tail -1) \$(free -g | awk '/^Mem:/{print \$7}') \$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -rn | head -1)" 2>/dev/null)
      free_kb=${free_kb:-0}; mem_g=${mem_g:-0}; gpu_max=${gpu_max:-999999}
      # Teardown-wait: if disk/RAM are fine but GPU is still busy, we very likely
      # just WON the flock from the previous holder whose server is still tearing
      # down (GPU frees in seconds). Hold the flock and poll up to ~40s for the
      # GPU to free instead of skipping -- else a less-careful neighbor grabs the
      # freshly-freed node while we back off. Bounded so an actively-busy node
      # (real neighbor eval) is skipped after ~40s.
      if [[ "$free_kb" -ge 1932735283 && "$mem_g" -ge 1300 && "$gpu_max" -ge 10000 ]]; then
        for _w in 1 2 3 4 5 6 7 8; do
          sleep 5
          gpu_max=$(srun --jobid="$jid" --overlap -N1 -w "$node" bash -c \
            "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -rn | head -1" 2>/dev/null)
          gpu_max=${gpu_max:-999999}
          [[ "$gpu_max" -lt 10000 ]] && break
        done
      fi
      if [[ "$free_kb" -ge 1932735283 && "$mem_g" -ge 1300 && "$gpu_max" -lt 10000 ]]; then
        echo "[run_eval] node $node OK (${mem_g}G RAM, $((free_kb/1024/1024))G disk, gpu ${gpu_max}MiB) -> running $VER"
        srun --jobid="$jid" --overlap -N1 -w "$node" --gres=gpu:8 bash "$EVAL" "$NAME" "$VER" "$@"
        rc=$?
        flock -u 200; exec 200>&-
        # Contamination check: a non-flock neighbor can collide mid-run (TOCTOU
        # after our GPU-idle gate), producing a slow, partial bench (seen: 63/7037,
        # 1497/7037). Treat <95% of 7037 completed as a failed (contaminated) run
        # and retry, so we don't log a garbage number.
        comp=$(grep -oiE "Successful requests: +[0-9]+" "$WORK/runs/$VER/mix.txt" 2>/dev/null | grep -oE "[0-9]+" | tail -1)
        comp=${comp:-0}
        if [ "$rc" -eq 0 ] && [ "$comp" -ge 6685 ]; then exit 0; fi   # 6685 ~= 0.95*7037
        attempts=$((attempts+1))
        if [ "$rc" -eq 0 ]; then
          echo "[run_eval] eval on $node CONTAMINATED (only $comp/7037 completed; likely collision) (attempt $attempts/$MAX_ATTEMPTS) -> retry"
        else
          echo "[run_eval] eval on $node FAILED rc=$rc (attempt $attempts/$MAX_ATTEMPTS) -> retry on another node"
        fi
        [ "$attempts" -ge "$MAX_ATTEMPTS" ] && { echo "[run_eval] giving up after $attempts attempts"; exit "${rc:-1}"; }
        sleep 30   # let a crashed/contaminated node settle before re-checking it
      else
        echo "[run_eval] node $node not ready (${mem_g}G RAM, $((free_kb/1024/1024))G disk, gpu ${gpu_max}MiB busy) -> skip"
        flock -u 200; exec 200>&-
      fi
    else
      exec 200>&-
    fi
  done
  echo "[run_eval] no free good-disk held node right now ($(date +%H:%M:%S)) — retry in 45s"
  sleep 8
done
