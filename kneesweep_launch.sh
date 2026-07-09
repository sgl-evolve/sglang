#!/usr/bin/env bash
# Grab a specific IDLE held certified node and run the FINE knee sweep for BOTH stock and cost-aware@2048 on it
# (same hardware, serial back-to-back) so the two p99-vs-λ curves near the knee are directly paired-comparable.
# Prefers an idle free-flock node (default ondem-3). Holds the flock for the whole run.
set -uo pipefail
POOL=/home/junyanch_google_com/autoresearch/workspace/sgl/v0.25_ablations/_pool
KS=/home/junyanch_google_com/autoresearch/workspace/sgl/v0.25_ablations/sgl_mech/researchers/sgl_mech/kneesweep.sh
PREF="${1:-slurm2-a3nodesetondem-3}"
RATES="${RATES:-3.0 3.6 4.0}"
TAGSUF="${TAGSUF:-}"   # e.g. "2" for the n=2 repeat -> knee-stock2 / knee-cost2
run(){
  local node="$1" jid="$2"
  echo "[ksL] === STOCK fine sweep on $node (rates: $RATES) tag=knee-stock$TAGSUF ==="
  srun --jobid="$jid" --overlap -N1 -w "$node" --gres=gpu:8 \
    --export=ALL,RATES="$RATES",SGLANG_ENABLE_COST_AWARE_EVICTION=0 \
    bash "$KS" "knee-stock$TAGSUF"
  echo "[ksL] stock rc=$?"
  echo "[ksL] === COST-AWARE@2048 fine sweep on $node (rates: $RATES) tag=knee-cost$TAGSUF ==="
  srun --jobid="$jid" --overlap -N1 -w "$node" --gres=gpu:8 \
    --export=ALL,RATES="$RATES",SGLANG_ENABLE_COST_AWARE_EVICTION=1,SGLANG_COST_AWARE_EVICT_THRESHOLD=2048,SGLANG_COST_AWARE_EVICT_THRESHOLD2=0,SGLANG_COST_AWARE_REUSE_MIN=0,SGLANG_COST_AWARE_COST_MODE=segment,SGLANG_ENABLE_COST_AWARE_MAMBA_EVICTION=0 \
    bash "$KS" "knee-cost$TAGSUF"
  echo "[ksL] cost rc=$?"
}
try_node(){
  local node="$1"
  local jid; jid=$(cat "$POOL/held/$node" 2>/dev/null) || return 1
  squeue -h -j "$jid" >/dev/null 2>&1 || return 1
  exec 200>"$POOL/locks/$node.lock"
  if flock -n 200; then
    echo "[ksL] acquired $node ($jid)"; run "$node" "$jid"
    flock -u 200; exec 200>&-; echo "[ksL] DONE released $node"; return 0
  fi
  exec 200>&-; return 1
}
# try preferred idle node first, then the other on-demand idle node, then any free held node
try_node "$PREF" && exit 0
for node in slurm2-a3nodesetondem-2 slurm2-a3nodesetondem-3 slurm2-a3nodeset0-3 slurm2-a3nodeset1-2; do
  [ "$node" = "$PREF" ] && continue
  try_node "$node" && exit 0
done
echo "[ksL] no free node acquired"; exit 1
