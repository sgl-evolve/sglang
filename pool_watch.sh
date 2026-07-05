#!/usr/bin/env bash
# Pool-aware eval watcher (correct path while a manager eval pool is live).
# Instead of self-locking a certified node (redundant + queue pressure when a pool exists), poll the
# manager's held pool nodes ($RT/held/*); when one has >=1800G free /mnt/localssd AND >=1300G RAM,
# acquire it COLLISION-SAFELY via the same per-node flock eval-on-pool.sh uses, wipe my own L3, and run
# my priority-ordered version list back-to-back on that one acquisition, then release. Highest-EV first:
#   v16 = --schedule-policy spf  (NEW aged shortest-prefill-first scheduler; mechanism)
#   v14 = --schedule-conservativeness 0.5   (config)
#   v15 = --enable-mixed-chunk              (config)
# The disk gate is why eval is currently blocked: both pool nodes are foreign-L3-full (<1800G). This
# watcher fires the instant that clears. Lightweight: one df probe per node per cycle.
set -uo pipefail
ROOT=/home/junyanch_google_com/autoresearch
set -a; . "$ROOT/.env"; set +a
SGL_HOME="${SGL_HOME:-$ROOT/programs/sgl}"
EVAL="$SGL_HOME/researcher/.claude/skills/evaluation-sop/scripts/eval.sh"
RT="${SGL_RUNTIME:-$SGL_HOME/manager/.runtime}"
NAME=kv-heron-eb9
CYCLE="${CYCLE:-600}"   # seconds between poll cycles

# If TARGET is set, only compete for those node(s) (skip persistently disk-short nodes so the flock
# retry on the recoverable node stays tight). Otherwise poll all held pool nodes.
held_nodes(){
  if [ -n "${TARGET:-}" ]; then for n in $TARGET; do [ -f "$RT/held/$n" ] && echo "$n"; done; return; fi
  [ -d "$RT/held" ] && for f in "$RT"/held/*; do [ -f "$f" ] && basename "$f"; done;
}

probe(){  # $1=node $2=holdjid -> "diskG ramG"
  timeout 60 srun --jobid="$2" --overlap -N1 -w "$1" bash -c \
    'd=$(df --output=avail -BG /mnt/localssd 2>/dev/null | tail -1 | tr -dc 0-9); m=$(awk "/MemAvailable/{print int(\$2/1024/1024)}" /proc/meminfo); echo ${d:-0} ${m:-0}' 2>/dev/null
}

gpu_max_mem(){  # $1=node $2=holdjid -> max GPU MiB used across the 8 GPUs (or "" on failure)
  timeout 60 srun --jobid="$2" --overlap -N1 -w "$1" bash -c \
    'nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | sort -rn | head -1' 2>/dev/null | tr -dc 0-9
}

run_seq(){  # $1=node $2=holdjid — run ONLY v16 (the critical mechanism) into this acquired hold.
  # Rationale: the pool is chaotically shared with a peer whose custom session doesn't take the flock,
  # so (a) hold the contended node ~40min not ~2h (fairer), and (b) get the ONE key result cleanly.
  local node="$1" jid="$2"
  echo "[poolw] wiping my L3 /mnt/localssd/$NAME on $node"
  srun --jobid="$jid" --overlap -N1 -w "$node" bash -c "rm -rf /mnt/localssd/$NAME/* 2>/dev/null; true" 2>/dev/null
  echo "[poolw] === running v16-be-spf on $node (pool hold $jid) ==="
  srun --jobid="$jid" --overlap -N1 -w "$node" --gres=gpu:8 bash "$EVAL" "$NAME" v16-be-spf \
    --enforce-disable-flashinfer-allreduce-fusion --hicache-storage-prefetch-policy best_effort --schedule-policy spf
  echo "[poolw] v16-be-spf rc=$?"
  srun --jobid="$jid" --overlap -N1 -w "$node" bash -c "rm -rf /mnt/localssd/$NAME/* 2>/dev/null; true" 2>/dev/null
}

echo "[poolw] starting; cycle=${CYCLE}s; gate=1800G disk / 1300G RAM; disk-short cooldown=${COOLDOWN:-300}s"
COOLDOWN="${COOLDOWN:-300}"
declare -A cooldown_until   # node -> loop-tick after which to re-probe a disk-short node
tick=0
while :; do
  ran=0
  tick=$((tick + 1))
  for node in $(held_nodes); do
    jid=$(cat "$RT/held/$node" 2>/dev/null) || continue
    # BLOCKING flock (-w): wait IN the kernel lock queue and acquire the instant the current holder
    # releases — this beats every non-blocking poller (eval-on-pool.sh uses `flock -n` on a 30s loop),
    # so I win the handoff on a heavily-contended node. Re-loop every BLOCKW seconds to re-evaluate.
    now_tick=$((tick * CYCLE))
    exec 200>"$RT/locks/$node.lock"
    if flock -w "${BLOCKW:-120}" 200; then
      if ! squeue -h -j "$jid" >/dev/null 2>&1; then
        flock -u 200; exec 200>&-; echo "[poolw] $node hold $jid dead — skip"; continue
      fi
      read -r diskg ramg < <(probe "$node" "$jid")
      gmem=$(gpu_max_mem "$node" "$jid")
      echo "[poolw] ACQUIRED-LOCK $node: disk=${diskg:-?}G ram=${ramg:-?}G gpu_max=${gmem:-?}MiB (hold $jid)"
      # GPU-IDLE gate: a peer's custom session may use this node WITHOUT taking the flock, so the flock
      # alone doesn't guarantee exclusivity. Only run if GPUs are genuinely idle (<2000 MiB used) — this
      # avoids colliding with (and being pkill'd by) a peer's concurrent server, and avoids thrashing a
      # node someone is actively using. Require a valid reading (empty = probe failed → treat as busy).
      if [ "${diskg:-0}" -ge 1800 ] && [ "${ramg:-0}" -ge 1300 ] && [ -n "${gmem:-}" ] && [ "${gmem:-999999}" -lt 2000 ]; then
        echo "[poolw] acquired $node (GPUs idle) — running v16"
        run_seq "$node" "$jid"
        flock -u 200; exec 200>&-
        echo "[poolw] DONE on $node"; ran=1; break
      else
        # not cleanly free (disk-short, or a peer is using the GPUs) — release and re-block, waiting
        # for a genuinely idle window so my eval doesn't collide.
        flock -u 200; exec 200>&-
        echo "[poolw] release $node (disk=${diskg:-?}G ram=${ramg:-?}G gpu=${gmem:-?}MiB not-clean) — re-block"
        sleep 5
      fi
    else
      exec 200>&-   # timed out waiting — re-loop (re-evaluate held_nodes, re-block)
    fi
  done
  [ "$ran" = 1 ] && { echo "[poolw] sequence complete — exiting watcher"; break; }
  sleep 1   # the real wait is the blocking flock -w above; re-block almost immediately on timeout
done
