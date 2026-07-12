#!/usr/bin/env bash
# OFF-CONTRACT DIAGNOSTIC (NOT the official eval) — boots the frozen 2-tier server and runs SHORT bench
# points to estimate prefill throughput P (server.log "input throughput") + hit-rate, to resolve the
# fork criterion EARLY on a non-certified idle node. Numbers are NOT certified-official; used only for
# the qualitative fork (is P ~4K [cold-bound/negative] or ~15K+ [cache-affectable]).
set -uo pipefail
ROOT=/home/junyanch_google_com/autoresearch
set -a; source "$ROOT/.env"; set +a
export HF_TOKEN="$HF_API_KEY" HF_HOME=/rmeng_data/junyanch-data/hf_cache HF_HUB_OFFLINE=1
WORK=/home/junyanch_google_com/autoresearch/workspace/sgl/v0.31/research/researchers/wilkes
source "$WORK/.venv/bin/activate"; export PYTHONPATH="$WORK/python"; cd "$WORK"
export TRITON_CACHE_DIR="$WORK/.cache/triton" CUDA_CACHE_PATH="$WORK/.cache/nv" FLASHINFER_CACHE_DIR="$WORK/.cache/flashinfer" XDG_CACHE_HOME="$WORK/.cache"; mkdir -p "$WORK/.cache"
MODEL=Qwen/Qwen3.5-122B-A10B-FP8; MIX=/rmeng_data/junyanch-data/datasets/mooncake_mix_v1.jsonl
OUT="$WORK/runs/diag"; mkdir -p "$OUT"; PORT=31007
export SGLANG_WILKES_TRACE="$OUT/trace"

DRAMG=$(awk '/MemAvailable/{printf "%d",$2/1048576}' /proc/meminfo); echo "DRAM ${DRAMG}G on $(hostname)"
[ "${DRAMG:-0}" -lt 1300 ] && { echo "DRAM_TOO_LOW"; exit 2; }

LAUNCH=(python3 -m sglang.launch_server --model-path "$MODEL" --tp 8 --trust-remote-code
  --context-length 262144 --chunked-prefill-size 6144 --mem-fraction-static 0.85
  --enable-hierarchical-cache --hicache-size 96 --page-size 64
  --hicache-io-backend direct --hicache-mem-layout page_first_direct --hicache-write-policy write_through
  --disable-custom-all-reduce --enforce-disable-flashinfer-allreduce-fusion
  --enable-metrics --enable-cache-report --dist-timeout 3600 --port "$PORT")
"${LAUNCH[@]}" > "$OUT/server.log" 2>&1 & SRV=$!
trap 'kill $SRV 2>/dev/null; sleep 4; pkill -9 -f sglang.launch_server 2>/dev/null' EXIT
for i in $(seq 1 900); do curl -sf localhost:$PORT/health >/dev/null 2>&1 && { echo "ready ~$((i*3))s"; break; }; kill -0 $SRV 2>/dev/null || { echo SERVER_DIED; tail -40 "$OUT/server.log"; exit 3; }; sleep 3; done
curl -sf localhost:$PORT/health >/dev/null 2>&1 || { echo SERVER_TIMEOUT; tail -40 "$OUT/server.log"; exit 3; }

# short warmup + two rate points (NUMP=400: enough for a P + hit estimate; NOT the pressured 1553)
for R in 3 7; do
  echo ">>> diag rate=$R $(date -u +%H:%M:%S)"
  python3 benchmark/hicache/bench_serving.py --backend sglang --model "$MODEL" \
    --dataset-name loogle --dataset-path "$MIX" --enable-multiturn --disable-shuffle \
    --request-rate "$R" --max-concurrency 256 --num-prompts 400 --port $PORT \
    --output-file "$OUT/bench_r$R.json" > "$OUT/bench_r$R.txt" 2>&1
  curl -s localhost:$PORT/metrics > "$OUT/metrics_r$R.txt" 2>/dev/null
  grep -E "P99 TTFT|Median TTFT|Request throughput|Output token throughput" "$OUT/bench_r$R.txt" | sed 's/^/    /'
done
echo "==== DIAG_DONE — server.log/metrics/trace in $OUT ===="
