#!/usr/bin/env bash
# FINE-GRAINED knee sweep (runs ON a held node via srun): ONE model load (FROZEN eval.sh launch flags), then
# bench the fixed mix at a FINE set of request rates around the p99=8s SLO knee to directly measure the
# goodput-under-SLO knee (the charter's PRIMARY metric) at a resolution my coarse λ∈{3,4,5,6} sweep lacked
# (it jumped λ=3 p99~5s -> λ=4 p99~8.5-10s). Only --request-rate varies; ALL launch flags stay frozen
# (identical to eval.sh). Cost-aware toggle comes from the environment (set via srun --export).
#
# CONTAMINATION CONTROL (so a clean p99 is possible even while siblings bench on other nodes):
#   - DeepGEMM JIT cache is put NODE-LOCAL (/mnt/localssd) -> my compiles never touch the shared NFS cache
#     that caused the historical parallel-eval p99 contamination (and I can't contaminate siblings either).
#   - RATES includes a λ=3.0 ANCHOR: known-clean p99 is ~5000-6000ms; if the anchor comes back inflated
#     (>8000ms) the run was contaminated and must be DISCARDED (don't log).
# Usage: kneesweep.sh <tag>   (RATES overridable via env; default "3.0 3.6 4.0")
set -uo pipefail
TAG="${1:?usage: kneesweep.sh <tag>}"
RATES="${RATES:-3.0 3.6 4.0}"
ROOT=/home/junyanch_google_com/autoresearch
WORK=/home/junyanch_google_com/autoresearch/workspace/sgl/v0.25_ablations/sgl_mech/researchers/sgl_mech
PORT=${PORT:-30000}
OUT="$WORK/runs/$TAG"; mkdir -p "$OUT"
set -a; source "$ROOT/.env"; set +a
export HF_TOKEN="$HF_API_KEY" HF_HOME=/rmeng_data/junyanch-data/hf_cache HF_HUB_OFFLINE=1
source "$WORK/.venv/bin/activate"; export PYTHONPATH="$WORK/python"; cd "$WORK"
export TRITON_CACHE_DIR="$WORK/.cache/triton" CUDA_CACHE_PATH="$WORK/.cache/nv" \
       FLASHINFER_CACHE_DIR="$WORK/.cache/flashinfer" XDG_CACHE_HOME="$WORK/.cache"
# NODE-LOCAL DeepGEMM cache -> isolate from shared NFS (no cross-eval p99 contamination in either direction)
export SGLANG_DG_CACHE_DIR=/mnt/localssd/sgl_mech_dg
mkdir -p "$WORK/.cache" "$SGLANG_DG_CACHE_DIR"
MODEL=Qwen/Qwen3.5-122B-A10B-FP8
MIX=/rmeng_data/junyanch-data/datasets/mooncake_mix_v1.jsonl
LAUNCH=(python3 -m sglang.launch_server
  --model-path "$MODEL" --tp 8 --trust-remote-code
  --context-length 262144 --chunked-prefill-size 6144 --mem-fraction-static 0.85
  --enable-hierarchical-cache --hicache-size 96 --page-size 64
  --hicache-io-backend direct --hicache-mem-layout page_first_direct --hicache-write-policy write_through
  --disable-custom-all-reduce --enforce-disable-flashinfer-allreduce-fusion
  --enable-metrics --enable-cache-report --port "$PORT")
echo ">>> [$TAG] launching server (cost_aware_evict=${SGLANG_ENABLE_COST_AWARE_EVICTION:-unset} thr=${SGLANG_COST_AWARE_EVICT_THRESHOLD:-unset}) DG=$SGLANG_DG_CACHE_DIR"
"${LAUNCH[@]}" > "$OUT/server.log" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null; sleep 5; pkill -9 -f sglang.launch_server 2>/dev/null' EXIT
for i in $(seq 1 1200); do   # allow up to ~60min (cold node-local DG JIT is slower)
  curl -sf "localhost:$PORT/health" >/dev/null 2>&1 && { echo "server ready (~$((i*3))s)"; break; }
  kill -0 $SRV 2>/dev/null || { echo "SERVER_DIED"; tail -40 "$OUT/server.log"; exit 3; }
  sleep 3
done
curl -sf "localhost:$PORT/health" >/dev/null 2>&1 || { echo "SERVER_TIMEOUT"; exit 3; }
grep -aE 'eviction strategy' "$OUT/server.log" | head -1
reset_cache(){ curl -s -X POST "localhost:$PORT/flush_cache" >/dev/null 2>&1; sleep 5; }
for R in $RATES; do
  echo ">>> [$TAG] bench @ lambda=$R ..."; reset_cache
  python3 benchmark/hicache/bench_serving.py --backend sglang --model "$MODEL" \
    --dataset-name loogle --dataset-path "$MIX" --enable-multiturn --disable-shuffle \
    --request-rate $R --max-concurrency 128 --num-prompts 1553 --port $PORT \
    --output-file "$OUT/rate_${R}.json" 2>&1 | tee "$OUT/rate_${R}.txt" | grep -E 'P99 TTFT|Median TTFT|Request throughput|Successful'
  echo "[$TAG] rate=$R done rc=${PIPESTATUS[0]}"
done
echo "[$TAG] KNEESWEEP_DONE"
