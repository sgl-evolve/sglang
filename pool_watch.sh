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

run_seq(){  # $1=node $2=holdjid — run the pre-registered CONFIG probes v14 then v15 into this hold.
  # v16 (--schedule-policy spf) is a CONFIRMED NEGATIVE (reliably hangs the server, 3 clean-node reps) —
  # dropped, never retried. v14/v15 do NOT change schedule-policy, so they won't hit the SPF hang; they
  # close the pre-registered scheduler-config path with real on-contract data. Each bounded by timeout so
  # a general server flake can't wedge the shared node; best-effort cleanup after each.
  local node="$1" jid="$2" ver args
  for spec in \
    "v14-be-cons0.5|--hicache-storage-prefetch-policy best_effort --schedule-conservativeness 0.5" \
    "v15-be-mixchunk|--hicache-storage-prefetch-policy best_effort --enable-mixed-chunk"; do
    ver="${spec%%|*}"; args="${spec#*|}"
    [ -f "runs/$ver/summary.json" ] && { echo "[poolw] $ver already done — skip"; continue; }
    echo "[poolw] wiping my L3 on $node; === running $ver ==="
    srun --jobid="$jid" --overlap -N1 -w "$node" bash -c "rm -rf /mnt/localssd/$NAME/* 2>/dev/null; true" 2>/dev/null
    timeout 2700 srun --jobid="$jid" --overlap -N1 -w "$node" --gres=gpu:8 bash "$EVAL" "$NAME" "$ver" \
      --enforce-disable-flashinfer-allreduce-fusion $args
    echo "[poolw] $ver rc=$?"
    srun --jobid="$jid" --overlap -N1 -w "$node" bash -c "pkill -9 -f 'sglang.launch_server.*$NAME' 2>/dev/null; rm -rf /mnt/localssd/$NAME/* 2>/dev/null; true" 2>/dev/null
  done
}

echo "[poolw] starting; cycle=${CYCLE}s; gate=1800G disk / 1300G RAM; disk-short cooldown=${COOLDOWN:-300}s"
COOLDOWN="${COOLDOWN:-300}"
IDLE_STREAK="${IDLE_STREAK:-4}"   # consecutive GPU-idle observations required = "peer truly done"
declare -A cooldown_until   # node -> loop-tick after which to re-probe a disk-short node
declare -A idle_count       # node -> consecutive cycles seen GPU-idle (distinguishes peer-done from between-evals)
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
      # SUSTAINED-GPU-IDLE gate: GPUs idle (<2GB) for IDLE_STREAK consecutive cycles => the peer is
      # genuinely DONE (not just briefly idle between its evals) => safe to run without colliding, and
      # robust to leftover dying/zombie procs (which hold 0 GPU). A single idle blip is NOT enough.
      if [ -n "${gmem:-}" ] && [ "${gmem:-999999}" -lt 2000 ]; then
        idle_count[$node]=$(( ${idle_count[$node]:-0} + 1 ))
      else
        idle_count[$node]=0
      fi
      echo "[poolw] $node: disk=${diskg:-?}G ram=${ramg:-?}G gpu_max=${gmem:-?}MiB procs=${gprocs:-?} idle_streak=${idle_count[$node]:-0}/$IDLE_STREAK (hold $jid)"
      if [ "${ramg:-0}" -ge 1300 ] && [ "${idle_count[$node]:-0}" -ge "$IDLE_STREAK" ]; then
        echo "[poolw] acquired $node (disk OK + sustained GPU-idle = peer done) — running config probes"
        run_seq "$node" "$jid"
        idle_count[$node]=0
        flock -u 200; exec 200>&-
        echo "[poolw] released $node after probe attempt"
      else
        flock -u 200; exec 200>&-
        echo "[poolw] release $node (ram=${ramg:-?}G gpu=${gmem:-?}MiB idle_streak=${idle_count[$node]:-0}) — retry"
      fi
    else
      exec 200>&-   # another holder — try next node / next cycle
    fi
  done
  # Exit only when BOTH config probes have real summaries; otherwise keep polling to retry the missing
  # one on a later clean window (v14/v15 don't hang like spf, so retrying is safe).
  if [ -f runs/v14-be-cons0.5/summary.json ] && [ -f runs/v15-be-mixchunk/summary.json ]; then
    echo "[poolw] both config probes complete — exiting watcher"; break
  fi
  sleep "${CYCLE:-10}"   # poll period for the non-blocking flock -n sweep of both pool nodes
done
