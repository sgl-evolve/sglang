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

run_seq(){  # $1=node $2=holdjid — run my version list into this acquired pool hold
  local node="$1" jid="$2"
  echo "[poolw] wiping my L3 /mnt/localssd/$NAME on $node"
  srun --jobid="$jid" --overlap -N1 -w "$node" bash -c "rm -rf /mnt/localssd/$NAME/* 2>/dev/null; true" 2>/dev/null
  local ver args
  for spec in \
    "v16-be-spf|--hicache-storage-prefetch-policy best_effort --schedule-policy spf" \
    "v14-be-cons0.5|--hicache-storage-prefetch-policy best_effort --schedule-conservativeness 0.5" \
    "v15-be-mixchunk|--hicache-storage-prefetch-policy best_effort --enable-mixed-chunk"; do
    ver="${spec%%|*}"; args="${spec#*|}"
    echo "[poolw] === running $ver on $node (pool hold $jid) ==="
    srun --jobid="$jid" --overlap -N1 -w "$node" --gres=gpu:8 bash "$EVAL" "$NAME" "$ver" \
      --enforce-disable-flashinfer-allreduce-fusion $args
    echo "[poolw] $ver rc=$?"
    # re-wipe between versions so a prior version's on-disk format can't poison the next
    srun --jobid="$jid" --overlap -N1 -w "$node" bash -c "rm -rf /mnt/localssd/$NAME/* 2>/dev/null; true" 2>/dev/null
  done
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
    # FLOCK-FIRST and PURELY LOCAL (no squeue per cycle → poll fast with zero slurmctld load).
    # Only after WINNING the flock do we verify the hold is alive + gate on disk/ram (rare path).
    now_tick=$((tick * CYCLE))
    exec 200>"$RT/locks/$node.lock"
    if flock -n 200; then
      if [ -n "${cooldown_until[$node]:-}" ] && [ "$now_tick" -lt "${cooldown_until[$node]}" ]; then
        flock -u 200; exec 200>&-; continue   # still disk-short (cooldown) — release fast, no srun
      fi
      if ! squeue -h -j "$jid" >/dev/null 2>&1; then
        flock -u 200; exec 200>&-; echo "[poolw] $node hold $jid dead — skip"; continue
      fi
      read -r diskg ramg < <(probe "$node" "$jid")
      echo "[poolw] LOCKED $node: disk=${diskg:-?}G ram=${ramg:-?}G (hold $jid)"
      if [ "${diskg:-0}" -ge 1800 ] && [ "${ramg:-0}" -ge 1300 ]; then
        echo "[poolw] acquired $node — running my version sequence"
        run_seq "$node" "$jid"
        flock -u 200; exec 200>&-
        echo "[poolw] DONE on $node"; ran=1; break
      else
        cooldown_until[$node]=$((now_tick + COOLDOWN))
        flock -u 200; exec 200>&-
        echo "[poolw] release $node (disk/ram ${diskg:-?}G/${ramg:-?}G) — cooldown ${COOLDOWN}s"
      fi
    else
      exec 200>&-   # another researcher holds it — retry next cycle (tight)
    fi
  done
  [ "$ran" = 1 ] && { echo "[poolw] sequence complete — exiting watcher"; break; }
  sleep "$CYCLE"
done
