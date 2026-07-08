#!/usr/bin/env bash
# same_node_goodput_ab.sh — definitive GOODPUT-KNEE A/B: on ONE held node, sweep baseline (fcfs) then
# exclusive across knee-region rates, holding the flock across BOTH so the p99-vs-load curves have NO
# node-to-node variance. Nails the headline metric (max req/s @ p99 TTFT <= 8s SLO).
set -uo pipefail
ROOT=/home/junyanch_google_com/autoresearch
POOL="$ROOT/workspace/sgl/v0.25_ablations/_pool"
WS="$ROOT/workspace/sgl/v0.25_ablations/sgl_free/researchers/sgl_free"
ONNODE="$WS/tools/sweep_onnode.sh"
RATES_KNEE="${RATES_KNEE:-3.5 4 4.5}"; NP="${NP:-600}"
BASE=snab_base; EXC=snab_exc
held_nodes(){ for f in "$POOL"/held/*; do [ -e "$f" ] && basename "$f"; done; }
echo "[gab] same-node goodput A/B: $BASE then $EXC, rates=[$RATES_KNEE] np=$NP"
for try in $(seq 1 200); do
  [ -f "$WS/runs/sweep_$EXC/curve.csv" ] && { echo "[gab] done"; break; }
  for node in $(held_nodes); do
    jid=$(cat "$POOL/held/$node" 2>/dev/null) || continue
    squeue -h -j "$jid" >/dev/null 2>&1 || continue
    exec 200>"$POOL/locks/$node.lock"
    if flock -n 200; then
      echo "[gab] try $try got $node (job $jid) $(date -u +%H:%M:%S)"
      unset SGLANG_HICACHE_EXCLUSIVE
      RATES="$RATES_KNEE" NPROMPTS="$NP" srun --jobid="$jid" --overlap -N1 -w "$node" --gres=gpu:8 \
        bash "$ONNODE" "$BASE"
      if [ ! -f "$WS/runs/sweep_$BASE/curve.csv" ]; then
        echo "[gab] baseline sweep failed on $node (DRAM?); release+retry"; flock -u 200; exec 200>&-; break
      fi
      export SGLANG_HICACHE_EXCLUSIVE=1
      RATES="$RATES_KNEE" NPROMPTS="$NP" srun --jobid="$jid" --overlap -N1 -w "$node" --gres=gpu:8 \
        bash "$ONNODE" "$EXC"
      flock -u 200; exec 200>&-
      [ -f "$WS/runs/sweep_$EXC/curve.csv" ] && { echo "[gab] SUCCESS both on $node"; break 2; }
      echo "[gab] exclusive sweep failed; retry"; break
    fi
    exec 200>&-
  done
  sleep 45
done
echo "[gab] === SAME-NODE GOODPUT CURVES ==="
echo "baseline:";  cat "$WS/runs/sweep_$BASE/curve.csv" 2>/dev/null
echo "exclusive:"; cat "$WS/runs/sweep_$EXC/curve.csv" 2>/dev/null
