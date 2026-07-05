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

fresh_check(){  # $1=node $2=holdjid -> "gpuMiB procN" ; empty/timeout => treat as not-fresh.
  # A node is "truly fresh" only if GPUs are idle AND no sglang.launch_server procs linger. A node I
  # previously wedged (D-state procs) fails this OR times out the srun -> skipped (won't re-hang there).
  timeout 60 srun --jobid="$2" --overlap -N1 -w "$1" bash -c \
    'g=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | sort -rn | head -1); p=$(pgrep -f sglang.launch_server 2>/dev/null | wc -l); echo ${g:-999999} ${p:-9}' 2>/dev/null
}

run_seq(){  # $1=node $2=holdjid — run ONLY v16 (the critical mechanism) into this acquired hold.
  # Rationale: the pool is chaotically shared with a peer whose custom session doesn't take the flock,
  # so (a) hold the contended node ~40min not ~2h (fairer), and (b) get the ONE key result cleanly.
  local node="$1" jid="$2"
  echo "[poolw] wiping my L3 /mnt/localssd/$NAME on $node"
  srun --jobid="$jid" --overlap -N1 -w "$node" bash -c "rm -rf /mnt/localssd/$NAME/* 2>/dev/null; true" 2>/dev/null
  echo "[poolw] === running v16-be-spf on $node (pool hold $jid) ==="
  # BOUNDED (timeout 2700s=45min): a real run is ~35min; if the server hangs (e.g. detokenizer stall
  # seen once), the timeout kills it rather than wedging the shared node indefinitely. Best-effort
  # server cleanup after, so a hung instance doesn't linger on the pool node.
  timeout 2700 srun --jobid="$jid" --overlap -N1 -w "$node" --gres=gpu:8 bash "$EVAL" "$NAME" v16-be-spf \
    --enforce-disable-flashinfer-allreduce-fusion --hicache-storage-prefetch-policy best_effort --schedule-policy spf
  echo "[poolw] v16-be-spf rc=$?"
  srun --jobid="$jid" --overlap -N1 -w "$node" bash -c "pkill -9 -f 'sglang.launch_server.*$NAME' 2>/dev/null; rm -rf /mnt/localssd/$NAME/* 2>/dev/null; true" 2>/dev/null
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
    now_tick=$((tick * CYCLE))
    # NON-BLOCKING flock -n so I poll BOTH pool nodes each cycle (1-2 AND ondem-3): whichever becomes
    # genuinely available first (flock-free + disk>=1800 + GPUs idle) wins. GPU-idle gate is essential
    # because a peer's custom session may use a node WITHOUT taking the flock. Disk-short nodes go on a
    # cooldown so I don't srun-probe them every cycle.
    exec 200>"$RT/locks/$node.lock"
    if flock -n 200; then
      if [ -n "${cooldown_until[$node]:-}" ] && [ "$now_tick" -lt "${cooldown_until[$node]}" ]; then
        flock -u 200; exec 200>&-; continue   # disk-short cooldown — skip probe
      fi
      if ! squeue -h -j "$jid" >/dev/null 2>&1; then
        flock -u 200; exec 200>&-; echo "[poolw] $node hold $jid dead — skip"; continue
      fi
      read -r diskg ramg < <(probe "$node" "$jid")
      if [ "${diskg:-0}" -lt 1800 ]; then
        cooldown_until[$node]=$((now_tick + COOLDOWN))
        flock -u 200; exec 200>&-
        echo "[poolw] $node disk-short (${diskg:-?}G) — cooldown ${COOLDOWN}s"; continue
      fi
      read -r gmem gprocs < <(fresh_check "$node" "$jid")
      echo "[poolw] $node: disk=${diskg:-?}G ram=${ramg:-?}G gpu_max=${gmem:-?}MiB sglang_procs=${gprocs:-?} (hold $jid)"
      # TRULY-FRESH gate: GPUs idle (<2GB) AND no lingering sglang procs (a node I previously wedged
      # keeps D-state procs and fails this / times out -> skipped, so I never re-hang on a damaged node).
      if [ "${ramg:-0}" -ge 1300 ] && [ -n "${gmem:-}" ] && [ "${gmem:-999999}" -lt 2000 ] && [ "${gprocs:-9}" -eq 0 ]; then
        echo "[poolw] acquired $node (disk OK + GPUs idle + no lingering procs = truly fresh) — running v16"
        run_seq "$node" "$jid"
        flock -u 200; exec 200>&-
        echo "[poolw] DONE on $node"; ran=1; break
      else
        flock -u 200; exec 200>&-
        echo "[poolw] release $node (ram=${ramg:-?}G gpu=${gmem:-?}MiB procs=${gprocs:-?} not-fresh) — retry"
      fi
    else
      exec 200>&-   # another holder — try next node / next cycle
    fi
  done
  [ "$ran" = 1 ] && { echo "[poolw] sequence complete — exiting watcher"; break; }
  sleep "${CYCLE:-10}"   # poll period for the non-blocking flock -n sweep of both pool nodes
done
