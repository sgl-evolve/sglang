#!/usr/bin/env bash
# sweep_onnode.sh <label> [extra server args...]   (run ON a compute node via srun; SGLANG_* env inherited)
# Goodput-curve SCREEN (off-curve, my call): ONE server load with the FROZEN protocol flags + extra args,
# then bench_serving at several request rates to find the knee (max req/s with p99 TTFT <= 8s SLO).
# Rates use a reduced num-prompts for speed (screen), warmed each rate. Reports p99 TTFT + req/s per rate.
set -uo pipefail
ROOT=/home/junyanch_google_com/autoresearch
WS=/home/junyanch_google_com/autoresearch/workspace/sgl/v0.25_ablations/sgl_free/researchers/sgl_free
set -a; source "$ROOT/.env"; set +a
export HF_TOKEN="$HF_API_KEY" HF_HOME=/rmeng_data/junyanch-data/hf_cache HF_HUB_OFFLINE=1
source "$WS/.venv/bin/activate"; export PYTHONPATH="$WS/python"; cd "$WS"
export TRITON_CACHE_DIR="$WS/.cache/triton" CUDA_CACHE_PATH="$WS/.cache/nv" \
       FLASHINFER_CACHE_DIR="$WS/.cache/flashinfer" XDG_CACHE_HOME="$WS/.cache"
LABEL="${1:?usage: sweep_onnode.sh <label> [extra args]}"; shift || true
EXTRA=("$@")
MODEL=Qwen/Qwen3.5-122B-A10B-FP8
MIX=/rmeng_data/junyanch-data/datasets/mooncake_mix_v1.jsonl
PORT=${PORT:-30055}
OUT="$WS/runs/sweep_$LABEL"; mkdir -p "$OUT"
RATES="${RATES:-3 4 5 6}"
NPROMPTS="${NPROMPTS:-800}"
echo "[sweep] label=$LABEL extra=[${EXTRA[*]:-none}] exclusive=[${SGLANG_HICACHE_EXCLUSIVE:-unset}] rates=[$RATES] nprompts=$NPROMPTS"

LAUNCH=(python3 -m sglang.launch_server --model-path "$MODEL" --tp 8 --trust-remote-code
  --context-length 262144 --chunked-prefill-size 6144 --mem-fraction-static 0.85
  --enable-hierarchical-cache --hicache-size 96 --page-size 64
  --hicache-io-backend direct --hicache-mem-layout page_first_direct --hicache-write-policy write_through
  --disable-custom-all-reduce --enforce-disable-flashinfer-allreduce-fusion
  --enable-metrics --enable-cache-report --port "$PORT")
[ ${#EXTRA[@]} -gt 0 ] && LAUNCH+=("${EXTRA[@]}")
"${LAUNCH[@]}" > "$OUT/server.log" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null; sleep 5; pkill -9 -f "port $PORT" 2>/dev/null' EXIT
for i in $(seq 1 900); do
  curl -sf "localhost:$PORT/health" >/dev/null 2>&1 && { echo "[sweep] server ready ~$((i*3))s"; break; }
  kill -0 $SRV 2>/dev/null || { echo "[sweep] SERVER_DIED"; tail -40 "$OUT/server.log"; exit 3; }
  sleep 3
done
echo "label,rate,p99_ttft_ms,p50_ttft_ms,req_throughput,completed" > "$OUT/curve.csv"
for R in $RATES; do
  curl -s -X POST "localhost:$PORT/flush_cache" >/dev/null 2>&1; sleep 5
  echo "[sweep] rate=$R ..."
  python3 benchmark/hicache/bench_serving.py --backend sglang --model "$MODEL" \
    --dataset-name loogle --dataset-path "$MIX" --enable-multiturn --disable-shuffle \
    --request-rate "$R" --max-concurrency 128 --num-prompts "$NPROMPTS" --port "$PORT" \
    --output-file "$OUT/mix_r$R.json" > "$OUT/bench_r$R.txt" 2>&1
  p99=$(grep -oE "P99 TTFT \(ms\):[[:space:]]*[0-9.]+" "$OUT/bench_r$R.txt" | grep -oE "[0-9.]+$" | tail -1)
  p50=$(grep -oE "Median TTFT \(ms\):[[:space:]]*[0-9.]+" "$OUT/bench_r$R.txt" | grep -oE "[0-9.]+$" | tail -1)
  rt=$(grep -oE "Request throughput \(req/s\):[[:space:]]*[0-9.]+" "$OUT/bench_r$R.txt" | grep -oE "[0-9.]+$" | tail -1)
  cp=$(grep -oE "Successful requests:[[:space:]]*[0-9]+" "$OUT/bench_r$R.txt" | grep -oE "[0-9]+$" | tail -1)
  echo "$LABEL,$R,${p99:-NA},${p50:-NA},${rt:-NA},${cp:-NA}" | tee -a "$OUT/curve.csv"
done
echo "[sweep] DONE -> $OUT/curve.csv"; cat "$OUT/curve.csv"
